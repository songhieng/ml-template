"""
PaDiM Anomaly Detection — Inference Module

This module provides a streamlined API for running inference with a trained PaDiM model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from scipy.spatial.distance import mahalanobis
from typing import Tuple, Optional
import joblib
import os

def _register_feature_hooks(model: torch.nn.Module, layer_names: list, features: dict):
    """Register forward hooks to capture features from specified ResNet layers."""
    def create_hook(name):
        def hook(module, input, output):
            features[name] = output.detach().cpu()
        return hook

    for name, module in model.named_children():
        if name in layer_names:
            module.register_forward_hook(create_hook(name))

class PaDiMInference:
    """Inference class for PaDiM anomaly detection."""
    def __init__(
        self,
        layers: list = ['layer1', 'layer2', 'layer3'],
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = layers
        self.features = {}

        # Load pre-trained ResNet18 backbone
        try:
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            self.model = models.resnet18(weights=None)

        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        _register_feature_hooks(self.model, self.layers, self.features)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.pca = None
        self.mean_vector = None
        self.covariance_inv = None
        self.threshold = None

    def load_model(self, model_path: str):
        """Load trained PCA, mean, covariance inverse, and threshold from .pkl or .npz."""
        if model_path.lower().endswith('.pkl'):
            data = joblib.load(model_path)
            self.pca = data['pca']
            self.mean_vector = data['mean_vector']
            self.covariance_inv = data['covariance_inv']
            self.threshold = data['threshold']

        elif model_path.lower().endswith('.npz'):
            data = np.load(model_path)
            if 'pca_components' in data:
                from sklearn.decomposition import PCA
                self.pca = PCA(n_components=data['pca_components'].shape[0])
                self.pca.components_ = data['pca_components']
                self.pca.mean_ = data['pca_mean']
                self.mean_vector = data['mean_vector']
                self.covariance_inv = data['covariance_inv']
                self.threshold = float(data['threshold'])
            else:
                raise ValueError("Unsupported NPZ format: expected 'pca_components' key.")
        else:
            raise ValueError("Unsupported model file format. Use .pkl or .npz")

    def _extract_features(self, image_path: str) -> np.ndarray:
        """Extract patch features from an image using ResNet layers."""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(tensor)

        feature_maps = []
        ref_size = self.features[self.layers[-1]].shape[2:]  # spatial dims of last layer

        for name in self.layers:
            fmap = self.features[name].squeeze(0)
            if fmap.shape[1:] != ref_size:
                fmap = F.interpolate(fmap.unsqueeze(0), size=ref_size,
                                     mode='bilinear', align_corners=False).squeeze(0)
            feature_maps.append(fmap)

        cat = torch.cat(feature_maps, dim=0)
        patches = cat.reshape(cat.shape[0], -1).T.cpu().numpy()
        return patches

    def infer(self, image_path: str) -> Tuple[float, str]:
        """Compute anomaly score and classification ('normal'/'anomaly') for a single image."""
        if self.pca is None:
            raise ValueError("Model not loaded. Call load_model() before inference.")

        patches = self._extract_features(image_path)
        reduced = self.pca.transform(patches)
        distances = np.array([
            mahalanobis(p, self.mean_vector, self.covariance_inv)
            for p in reduced
        ])
        score = float(distances.max())
        label = 'anomaly' if score > self.threshold else 'normal'
        return score, label

def run_inference(image_path: str, model_path: str) -> dict:
    """Convenience function to run inference and return results as a dict."""
    inf = PaDiMInference()
    inf.load_model(model_path)
    score, label = inf.infer(image_path)
    return {
        'image_path': image_path,
        'anomaly_score': score,
        'classification': label,
        'is_anomaly': (label == 'anomaly')
    }