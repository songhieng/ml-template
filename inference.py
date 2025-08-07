#!/usr/bin/env python3
"""
Recaptured Image Detection Inference Module

This module provides inference capabilities for the recaptured image detection model.
It can classify images as either Natural Images (NI) or Recaptured Images (RI).

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --batch path/to/image/folder
    python inference.py --image path/to/image.jpg --output results.json
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import datetime
from model import MrcNet

class RecaptureDetector:
    def __init__(self, model_path="models/checkpoint_100.pth", device=None):
        """
        Initialize the recapture detector.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            device (str): Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MrcNet()
        self.load_model()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to a standard size
            transforms.CenterCrop(96),      # Center crop to patch size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Class labels
        self.class_names = {0: 'RI (Recaptured Image)', 1: 'NI (Natural Image)'}
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model checkpoint: {self.model_path}")
    
    def load_model(self):
        """Load the trained model from checkpoint."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for inference.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            raise Exception(f"Failed to preprocess image {image_path}: {str(e)}")
    
    def predict_single(self, image_path, return_probabilities=True):
        """
        Predict whether a single image is recaptured or natural.
        
        Args:
            image_path (str): Path to the image file
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results containing class, confidence, and probabilities
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            result = {
                'image_path': str(image_path),
                # 'predicted_class': predicted_class,
                # 'class_name': self.class_names[predicted_class],
                'confidence': float(confidence_score),
                'is_recaptured': predicted_class == 1,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'RI_probability': float(probabilities[0][1]),
                    'NI_probability': float(probabilities[0][0])
                }
            
            return result
    
    def predict_batch(self, image_folder, output_file=None, return_probabilities=True):
        """
        Predict for all images in a folder.
        
        Args:
            image_folder (str): Path to folder containing images
            output_file (str): Path to save results JSON file (optional)
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results for all images
        """
        image_folder = Path(image_folder)
        if not image_folder.exists():
            raise Exception(f"Image folder does not exist: {image_folder}")
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_folder.glob(f"*{ext}")))
            image_files.extend(list(image_folder.glob(f"*{ext.upper()}")))
        
        if not image_files:
            raise Exception(f"No image files found in {image_folder}")
        
        print(f"Found {len(image_files)} images to process...")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            try:
                print(f"Processing {i}/{len(image_files)}: {image_file.name}")
                result = self.predict_single(image_file, return_probabilities)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                results.append({
                    'image_path': str(image_file),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results, output_file):
        """
        Save prediction results to a JSON file.
        
        Args:
            results (list): List of prediction results
            output_file (str): Path to output JSON file
        """
        output_path = Path("output_results") / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        # Create summary statistics
        successful_predictions = [r for r in results if 'error' not in r]
        total_images = len(results)
        successful_count = len(successful_predictions)
        
        if successful_predictions:
            recaptured_count = sum(1 for r in successful_predictions if r.get('is_recaptured', False))
            natural_count = successful_count - recaptured_count
            avg_confidence = np.mean([r['confidence'] for r in successful_predictions])
        else:
            recaptured_count = natural_count = avg_confidence = 0
        
        summary = {
            'total_images': total_images,
            'successful_predictions': successful_count,
            'failed_predictions': total_images - successful_count,
            'recaptured_images': recaptured_count,
            'natural_images': natural_count,
            'average_confidence': float(avg_confidence),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        output_data = {
            'summary': summary,
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        print(f"Summary: {recaptured_count} recaptured, {natural_count} natural images")
    
    def print_summary(self, results):
        """Print a summary of prediction results."""
        successful_predictions = [r for r in results if 'error' not in r]
        
        if not successful_predictions:
            print("No successful predictions to summarize.")
            return
        
        recaptured_count = sum(1 for r in successful_predictions if r.get('is_recaptured', False))
        natural_count = len(successful_predictions) - recaptured_count
        avg_confidence = np.mean([r['confidence'] for r in successful_predictions])
        
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Total images processed: {len(results)}")
        print(f"Successful predictions: {len(successful_predictions)}")
        print(f"Recaptured images (RI): {recaptured_count}")
        print(f"Natural images (NI): {natural_count}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Recaptured Image Detection Inference')
    parser.add_argument('--image', type=str, help='Path to a single image file')
    parser.add_argument('--batch', type=str, help='Path to folder containing images')
    parser.add_argument('--model', type=str, default='models/checkpoint_100.pth',
                        help='Path to model checkpoint (default: models/checkpoint_100.pth)')
    parser.add_argument('--output', type=str, help='Output JSON file name (saved in output_results/)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], 
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--no-probabilities', action='store_true',
                        help='Don\'t include class probabilities in output')
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        parser.error("Either --image or --batch must be specified")
    
    # Initialize detector
    try:
        detector = RecaptureDetector(model_path=args.model, device=args.device)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Process single image
    if args.image:
        try:
            result = detector.predict_single(
                args.image, 
                return_probabilities=not args.no_probabilities
            )
            
            print("\n" + "="*50)
            print("SINGLE IMAGE PREDICTION")
            print("="*50)
            print(f"Image: {result['image_path']}")
            print(f"Prediction: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if 'probabilities' in result:
                print(f"RI Probability: {result['probabilities']['RI_probability']:.3f}")
                print(f"NI Probability: {result['probabilities']['NI_probability']:.3f}")
            
            print("="*50)
            
            # Save single result if output specified
            if args.output:
                detector.save_results([result], args.output)
                
        except Exception as e:
            print(f"Error processing image: {e}")
    
    # Process batch of images
    if args.batch:
        try:
            results = detector.predict_batch(
                args.batch,
                output_file=args.output,
                return_probabilities=not args.no_probabilities
            )
            
            detector.print_summary(results)
            
        except Exception as e:
            print(f"Error processing batch: {e}")


if __name__ == "__main__":
    main() 
