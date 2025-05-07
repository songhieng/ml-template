import os
import shutil
import argparse
from PIL import Image, ImageOps
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

def estimate_depth(input_dir, output_dir, model_name="Intel/dpt-hybrid-midas", device=None):
    """
    Run depth estimation on all images in `input_dir` and save depth maps to `output_dir`,
    preserving subfolder structure.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    feat = DPTFeatureExtractor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name).to(device)

    for root, _, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        tgt = os.path.join(output_dir, rel)
        os.makedirs(tgt, exist_ok=True)

        for fname in tqdm(files, desc=f"Depth {rel}"):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img = Image.open(os.path.join(root, fname)).convert("RGB")
            inputs = feat(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs).predicted_depth

            depth = torch.nn.functional.interpolate(
                outputs.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

            d_min, d_max = depth.min(), depth.max()
            norm = (depth - d_min) / (d_max - d_min + 1e-8)
            depth_img = Image.fromarray((norm * 255).astype(np.uint8))
            depth_img.save(os.path.join(tgt, fname))


def augment_images(input_dir, output_dir, n_augment=3):
    """
    Simple augmentation by mirroring each image `n_augment` times.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for fname in tqdm(files, desc=f"Augment {os.path.basename(input_dir)}"):
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        # save original as well
        base, ext = os.path.splitext(fname)
        img.save(os.path.join(output_dir, fname))
        for i in range(n_augment):
            aug = ImageOps.mirror(img)
            aug.save(os.path.join(output_dir, f"{base}_aug{i}{ext}"))


def build_model(device):
    """
    Instantiate a ResNet18-based binary classifier with dropout + sigmoid head.
    """
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 1),
        nn.Sigmoid()
    )
    return model.to(device)


def train_and_validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Depth estimation
    depth_dir = os.path.join(args.data_dir, "depth")
    estimate_depth(args.data_dir, depth_dir, device=device)

    # 2) Processed directory setup
    proc_dir   = os.path.join(args.data_dir, "processed")
    df_src     = os.path.join(depth_dir, "deepfake")
    df_dst     = os.path.join(proc_dir,   "deepfake")
    real_src   = os.path.join(depth_dir, "real")
    real_dst   = os.path.join(proc_dir,   "real")

    os.makedirs(df_dst, exist_ok=True)
    os.makedirs(real_dst, exist_ok=True)

    # 2a) Augment only deepfake
    augment_images(df_src, df_dst, args.n_augment)

    # 2b) Copy real images untouched
    for fname in os.listdir(real_src):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.copy(
                os.path.join(real_src, fname),
                os.path.join(real_dst, fname)
            )

    # 3) Transforms and dataset split
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    full_ds = datasets.ImageFolder(proc_dir, transform=train_tf)
    n_val   = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # 4) Model, loss, optimizer, scheduler
    model     = build_model(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} ▶"):
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} ◀"):
                imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                preds = torch.round(outputs).cpu().numpy().flatten()
                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(preds)

        avg_val = val_loss / len(val_loader)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec  = recall_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        print(f"Val  Loss: {avg_val:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        scheduler.step(avg_val)
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), args.ckpt)
            print("👉 Saved new best model checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFake Classifier (auto-validation)")
    parser.add_argument("--data_dir",   type=str,   default="data/",
                        help="Root folder with 'deepfake' & 'real' subfolders of raw RGB images")
    parser.add_argument("--ckpt",       type=str,   default="best.pth",
                        help="Path to save the best model checkpoint")
    parser.add_argument("--epochs",     type=int,   default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Initial learning rate for Adam optimizer")
    parser.add_argument("--n_augment",  type=int,   default=3,
                        help="How many mirrored augmentations per image")
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Fraction of data to hold out for validation (e.g. 0.2 = 20%)")
    args = parser.parse_args()

    train_and_validate(args)
