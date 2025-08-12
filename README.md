"""
PaDiM Anomaly Detection Module

A professional implementation of PaDiM (Patch Distribution Modeling) for anomaly detection.
This module provides a clean API for anomaly detection in images using pre-trained ResNet features.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import joblib


def _register_feature_hooks(model, layer_names: list, features_dict: dict):
    """Register forward hooks to extract features from specified layers."""
    def create_hook(name):
        def hook(module, input, output):
            features_dict[name] = output.detach().cpu()
        return hook

    for name, module in model.named_children():
        if name in layer_names:
            module.register_forward_hook(create_hook(name))


class PaDiMDetector:
    """
    PaDiM anomaly detector for identifying anomalous regions in images.
    
    This implementation uses ResNet backbone to extract multi-scale features,
    applies PCA for dimensionality reduction, and models normal patches with
    a multivariate Gaussian distribution.
    """
    
    def __init__(self, 
                 layers: list = ['layer1', 'layer2', 'layer3'],
                 pca_components: int = 64,
                 device: Optional[torch.device] = None):
        """
        Initialize PaDiM detector.
        
        Args:
            layers: ResNet layer names to extract features from
            pca_components: Number of PCA components for dimensionality reduction
            device: PyTorch device (auto-detected if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = layers
        self.pca_components = pca_components
        
        # Initialize ResNet backbone using explicit weights enum for compatibility
        try:
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            # Fallback without weights if enum not available or weights cannot be resolved
            self.model = models.resnet18(weights=None)
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Setup feature extraction hooks
        self.features = {}
        _register_feature_hooks(self.model, self.layers, self.features)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Model parameters (set during training)
        self.pca = None
        self.mean_vector = None
        self.covariance_inv = None
        self.threshold = None

    def _extract_patch_features(self, image_path: str) -> np.ndarray:
        """
        Extract multi-scale patch features from an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Array of patch features with shape (num_patches, feature_dim)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            _ = self.model(tensor)
        
        # Collect and align features from different layers
        feature_maps = []
        reference_size = self.features[self.layers[-1]].shape[2:]  # Use last layer as reference
        
        for layer_name in self.layers:
            feature_map = self.features[layer_name].squeeze(0)
            
            # Resize to match reference size if needed
            if feature_map.shape[1:] != reference_size:
                feature_map = F.interpolate(
                    feature_map.unsqueeze(0), 
                    size=reference_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            feature_maps.append(feature_map)
        
        # Concatenate features and reshape to patches
        concatenated = torch.cat(feature_maps, dim=0)
        patches = concatenated.reshape(concatenated.shape[0], -1).T.cpu().numpy()
        
        return patches
    
    def load(self, filepath: str):
        """
        Load trained model parameters.
        
        Args:
            filepath: Path to saved model file
        """
        if filepath.lower().endswith('.pkl'):
            data = joblib.load(filepath)
            self.pca = data['pca']
            self.mean_vector = data['mean_vector']
            self.covariance_inv = data['covariance_inv']
            self.threshold = data['threshold']
            
        elif filepath.lower().endswith('.npz'):
            data = np.load(filepath)
            
            # Handle both new format and existing model format
            if 'pca_components' in data:
                # New format created by our refactored code
                self.pca = PCA(n_components=data['pca_components'].shape[0])
                self.pca.components_ = data['pca_components']
                self.pca.mean_ = data['pca_mean']
                
                self.mean_vector = data['mean_vector']
                self.covariance_inv = data['covariance_inv']
                self.threshold = float(data['threshold'])
                
            elif 'mus' in data and 'cov_invs' in data:
                # Existing/legacy model format - create adapter
                # print("📦 Loading existing PaDiM model format...")  # Suppressed for API mode
                
                # Extract parameters from existing format
                mus = data['mus']              # Mean vectors for each spatial location
                cov_invs = data['cov_invs']    # Inverse covariance matrices 
                self.threshold = float(data['thresh'])
                
                # Get dimensions
                H, W, D = int(data['H']), int(data['W']), int(data['D'])
                
                # Store legacy format parameters
                self.legacy_mus = mus
                self.legacy_cov_invs = cov_invs
                self.legacy_H = H
                self.legacy_W = W
                self.legacy_D = D
                
                # Extract channel indices if available
                if 'ch_idx' in data:
                    self.legacy_ch_idx = data['ch_idx']
                else:
                    # Default channel selection for ResNet layers
                    self.legacy_ch_idx = np.arange(D)
                
                # Set flag to use legacy prediction method
                self.use_legacy_format = True
                
                # print(f"   Loaded legacy model: {H}x{W} spatial, {D}D features")  # Suppressed for API mode
                # print(f"   Threshold: {self.threshold:.3f}")  # Suppressed for API mode
                
            else:
                # Unknown format
                print("❌ Unknown model format.")
                print("    Available keys:", list(data.keys()))
                raise ValueError("Unknown model format. Expected either new format with 'pca_components' or legacy format with 'mus'.")
            
        else:
            raise ValueError("Unsupported format. Use .pkl or .npz")
        
        # print(f"Model loaded from: {filepath}")  # Suppressed for API mode
    
    def _predict_legacy(self, image_path: str) -> Tuple[float, str]:
        """
        Predict using legacy model format.
        
        Args:
            image_path: Path to test image
            
        Returns:
            Tuple of (anomaly_score, classification)
        """
        # Use a simpler feature extraction approach for legacy compatibility
        # Load and preprocess image to match original training
        image = Image.open(image_path).convert('RGB')
        
        # Resize to standard input size that matches training
        resized_image = image.resize((224, 224), Image.LANCZOS)
        
        # Convert to tensor format
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        tensor = transform(resized_image).unsqueeze(0).to(self.device)
        
        # Extract features using a different approach for legacy compatibility
        with torch.no_grad():
            _ = self.model(tensor)
        
        # Get features and resize to match expected dimensions
        feature_maps = []
        target_size = (self.legacy_H, self.legacy_W)  # 28x28
        
        for layer_name in self.layers:
            if layer_name in self.features:
                feature_map = self.features[layer_name].squeeze(0)
                
                # Resize to target spatial dimensions
                if feature_map.shape[1:] != target_size:
                    feature_map = F.interpolate(
                        feature_map.unsqueeze(0), 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                
                feature_maps.append(feature_map)
        
        if not feature_maps:
            print("⚠️  Warning: No feature maps extracted. Using default values.")
            # Return a default score that's below threshold
            return 0.0, 'normal'
        
        # Concatenate features
        concatenated = torch.cat(feature_maps, dim=0)
        patches = concatenated.reshape(concatenated.shape[0], -1).T.cpu().numpy()
        
        # print(f"   Extracted patches shape: {patches.shape}")  # Suppressed for API mode
        # print(f"   Legacy model expects: {self.legacy_H}x{self.legacy_W} = {self.legacy_H * self.legacy_W} patches, {self.legacy_D}D features")  # Suppressed for API mode
        
        # Handle feature dimension mismatch more robustly
        available_features = patches.shape[1]
        if available_features != self.legacy_D:
            if available_features < self.legacy_D:
                # Repeat features to reach target dimension
                repeat_factor = self.legacy_D // available_features
                remainder = self.legacy_D % available_features
                
                repeated_features = np.tile(patches, (1, repeat_factor))
                if remainder > 0:
                    additional_features = patches[:, :remainder]
                    patches = np.hstack([repeated_features, additional_features])
                else:
                    patches = repeated_features
            else:
                # Use PCA or select most important features
                # For simplicity, take first D features
                patches = patches[:, :self.legacy_D]
        
        # Ensure we have the right number of spatial locations
        num_patches = patches.shape[0]
        expected_patches = self.legacy_H * self.legacy_W
        
        if num_patches != expected_patches:
            if num_patches < expected_patches:
                # Upsample by repeating patches
                repeat_indices = np.tile(np.arange(num_patches), 
                                       (expected_patches // num_patches + 1))[:expected_patches]
                patches = patches[repeat_indices]
            else:
                # Downsample by selecting evenly spaced patches
                indices = np.linspace(0, num_patches - 1, expected_patches, dtype=int)
                patches = patches[indices]
        
        # print(f"   Final patches shape: {patches.shape}")  # Suppressed for API mode
        
        # Compute Mahalanobis distances
        distances = []
        successful_computations = 0
        
        for i in range(self.legacy_H * self.legacy_W):
            try:
                patch = patches[i]
                mu = self.legacy_mus[i]
                cov_inv = self.legacy_cov_invs[i]
                
                # Ensure dimensions match exactly
                if len(patch) != len(mu):
                    # This shouldn't happen now, but handle gracefully
                    if len(patch) < len(mu):
                        patch = np.pad(patch, (0, len(mu) - len(patch)), 'constant')
                    else:
                        patch = patch[:len(mu)]
                
                # Compute Mahalanobis distance with numerical stability
                diff = patch - mu
                try:
                    distance = np.sqrt(np.abs(diff.T @ cov_inv @ diff))
                    distances.append(distance)
                    successful_computations += 1
                except np.linalg.LinAlgError:
                    # Handle singular matrix case
                    distances.append(np.linalg.norm(diff))
                    
            except Exception as e:
                distances.append(0.0)
        
        # print(f"   Successfully computed distances for {successful_computations}/{len(distances)} locations")  # Suppressed for API mode
        
        # Use maximum distance as anomaly score
        if len(distances) > 0:
            anomaly_score = float(np.max(distances))
        else:
            anomaly_score = 0.0
            
        classification = 'anomaly' if anomaly_score > self.threshold else 'normal'

        return anomaly_score, classification

    def predict(self, image_path: str) -> Tuple[float, str]:
        """
        Predict anomaly score and classification for an image.
        
        Args:
            image_path: Path to test image

        Returns:
            Tuple of (anomaly_score, classification)
            - anomaly_score: Maximum Mahalanobis distance across patches
            - classification: 'normal' or 'anomaly' based on threshold
        """
        if hasattr(self, 'use_legacy_format') and self.use_legacy_format:
            return self._predict_legacy(image_path)
        
        # Standard prediction method for new format
        if self.pca is None:
            raise ValueError("Model not trained. Call train() first or load() a trained model.")
        
        # Extract and reduce patch features
        patches = self._extract_patch_features(image_path)
        reduced_patches = self.pca.transform(patches)
        
        # Compute Mahalanobis distances
        distances = np.array([
            mahalanobis(patch, self.mean_vector, self.covariance_inv)
            for patch in reduced_patches
        ])
        
        # Use maximum distance as anomaly score
        anomaly_score = distances.max()
        classification = 'anomaly' if anomaly_score > self.threshold else 'normal'
        
        return anomaly_score, classification


def detect_anomaly(image_path: str, model_path: str) -> dict:
    """
    Convenience function for anomaly detection on a single image.
    
    Args:
        image_path: Path to test image
        model_path: Path to trained PaDiM model
        
    Returns:
        Dictionary containing detection results
    """
    detector = PaDiMDetector()
    detector.load(model_path)
    score, label = detector.predict(image_path)
    
    return {
        'image_path': str(image_path),
        'anomaly_score': score,
        'classification': label,
        'is_anomaly': label == 'anomaly'
    }


    the above code is the training code and the below code is the inference code padim module that we are using for the inference for our system can you write anoter file of the padim infereence module as before for the above training code please: 
"""
PaDiM Anomaly Detection Module

A professional implementation of PaDiM (Patch Distribution Modeling) for anomaly detection.
This module provides a clean API for anomaly detection in images using pre-trained ResNet features.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import joblib


def _register_feature_hooks(model, layer_names: list, features_dict: dict):
    """Register forward hooks to extract features from specified layers."""
    def create_hook(name):
        def hook(module, input, output):
            features_dict[name] = output.detach().cpu()
        return hook

    for name, module in model.named_children():
        if name in layer_names:
            module.register_forward_hook(create_hook(name))


class PaDiMDetector:
    """
    PaDiM anomaly detector for identifying anomalous regions in images.
    
    This implementation uses ResNet backbone to extract multi-scale features,
    applies PCA for dimensionality reduction, and models normal patches with
    a multivariate Gaussian distribution.
    """
    
    def __init__(self, 
                 layers: list = ['layer1', 'layer2', 'layer3'],
                 pca_components: int = 64,
                 device: Optional[torch.device] = None):
        """
        Initialize PaDiM detector.
        
        Args:
            layers: ResNet layer names to extract features from
            pca_components: Number of PCA components for dimensionality reduction
            device: PyTorch device (auto-detected if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = layers
        self.pca_components = pca_components
        
        # Initialize ResNet backbone using explicit weights enum for compatibility
        try:
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            # Fallback without weights if enum not available or weights cannot be resolved
            self.model = models.resnet18(weights=None)
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Setup feature extraction hooks
        self.features = {}
        _register_feature_hooks(self.model, self.layers, self.features)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Model parameters (set during training)
        self.pca = None
        self.mean_vector = None
        self.covariance_inv = None
        self.threshold = None

    def _extract_patch_features(self, image_path: str) -> np.ndarray:
        """
        Extract multi-scale patch features from an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Array of patch features with shape (num_patches, feature_dim)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            _ = self.model(tensor)
        
        # Collect and align features from different layers
        feature_maps = []
        reference_size = self.features[self.layers[-1]].shape[2:]  # Use last layer as reference
        
        for layer_name in self.layers:
            feature_map = self.features[layer_name].squeeze(0)
            
            # Resize to match reference size if needed
            if feature_map.shape[1:] != reference_size:
                feature_map = F.interpolate(
                    feature_map.unsqueeze(0), 
                    size=reference_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            feature_maps.append(feature_map)
        
        # Concatenate features and reshape to patches
        concatenated = torch.cat(feature_maps, dim=0)
        patches = concatenated.reshape(concatenated.shape[0], -1).T.cpu().numpy()
        
        return patches
    
    def load(self, filepath: str):
        """
        Load trained model parameters.
        
        Args:
            filepath: Path to saved model file
        """
        if filepath.lower().endswith('.pkl'):
            data = joblib.load(filepath)
            self.pca = data['pca']
            self.mean_vector = data['mean_vector']
            self.covariance_inv = data['covariance_inv']
            self.threshold = data['threshold']
            
        elif filepath.lower().endswith('.npz'):
            data = np.load(filepath)
            
            # Handle both new format and existing model format
            if 'pca_components' in data:
                # New format created by our refactored code
                self.pca = PCA(n_components=data['pca_components'].shape[0])
                self.pca.components_ = data['pca_components']
                self.pca.mean_ = data['pca_mean']
                
                self.mean_vector = data['mean_vector']
                self.covariance_inv = data['covariance_inv']
                self.threshold = float(data['threshold'])
                
            elif 'mus' in data and 'cov_invs' in data:
                # Existing/legacy model format - create adapter
                # print("📦 Loading existing PaDiM model format...")  # Suppressed for API mode
                
                # Extract parameters from existing format
                mus = data['mus']              # Mean vectors for each spatial location
                cov_invs = data['cov_invs']    # Inverse covariance matrices 
                self.threshold = float(data['thresh'])
                
                # Get dimensions
                H, W, D = int(data['H']), int(data['W']), int(data['D'])
                
                # Store legacy format parameters
                self.legacy_mus = mus
                self.legacy_cov_invs = cov_invs
                self.legacy_H = H
                self.legacy_W = W
                self.legacy_D = D
                
                # Extract channel indices if available
                if 'ch_idx' in data:
                    self.legacy_ch_idx = data['ch_idx']
                else:
                    # Default channel selection for ResNet layers
                    self.legacy_ch_idx = np.arange(D)
                
                # Set flag to use legacy prediction method
                self.use_legacy_format = True
                
                # print(f"   Loaded legacy model: {H}x{W} spatial, {D}D features")  # Suppressed for API mode
                # print(f"   Threshold: {self.threshold:.3f}")  # Suppressed for API mode
                
            else:
                # Unknown format
                print("❌ Unknown model format.")
                print("    Available keys:", list(data.keys()))
                raise ValueError("Unknown model format. Expected either new format with 'pca_components' or legacy format with 'mus'.")
            
        else:
            raise ValueError("Unsupported format. Use .pkl or .npz")
        
        # print(f"Model loaded from: {filepath}")  # Suppressed for API mode
    
    def _predict_legacy(self, image_path: str) -> Tuple[float, str]:
        """
        Predict using legacy model format.
        
        Args:
            image_path: Path to test image
            
        Returns:
            Tuple of (anomaly_score, classification)
        """
        # Use a simpler feature extraction approach for legacy compatibility
        # Load and preprocess image to match original training
        image = Image.open(image_path).convert('RGB')
        
        # Resize to standard input size that matches training
        resized_image = image.resize((224, 224), Image.LANCZOS)
        
        # Convert to tensor format
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        tensor = transform(resized_image).unsqueeze(0).to(self.device)
        
        # Extract features using a different approach for legacy compatibility
        with torch.no_grad():
            _ = self.model(tensor)
        
        # Get features and resize to match expected dimensions
        feature_maps = []
        target_size = (self.legacy_H, self.legacy_W)  # 28x28
        
        for layer_name in self.layers:
            if layer_name in self.features:
                feature_map = self.features[layer_name].squeeze(0)
                
                # Resize to target spatial dimensions
                if feature_map.shape[1:] != target_size:
                    feature_map = F.interpolate(
                        feature_map.unsqueeze(0), 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                
                feature_maps.append(feature_map)
        
        if not feature_maps:
            print("⚠️  Warning: No feature maps extracted. Using default values.")
            # Return a default score that's below threshold
            return 0.0, 'normal'
        
        # Concatenate features
        concatenated = torch.cat(feature_maps, dim=0)
        patches = concatenated.reshape(concatenated.shape[0], -1).T.cpu().numpy()
        
        # print(f"   Extracted patches shape: {patches.shape}")  # Suppressed for API mode
        # print(f"   Legacy model expects: {self.legacy_H}x{self.legacy_W} = {self.legacy_H * self.legacy_W} patches, {self.legacy_D}D features")  # Suppressed for API mode
        
        # Handle feature dimension mismatch more robustly
        available_features = patches.shape[1]
        if available_features != self.legacy_D:
            if available_features < self.legacy_D:
                # Repeat features to reach target dimension
                repeat_factor = self.legacy_D // available_features
                remainder = self.legacy_D % available_features
                
                repeated_features = np.tile(patches, (1, repeat_factor))
                if remainder > 0:
                    additional_features = patches[:, :remainder]
                    patches = np.hstack([repeated_features, additional_features])
                else:
                    patches = repeated_features
            else:
                # Use PCA or select most important features
                # For simplicity, take first D features
                patches = patches[:, :self.legacy_D]
        
        # Ensure we have the right number of spatial locations
        num_patches = patches.shape[0]
        expected_patches = self.legacy_H * self.legacy_W
        
        if num_patches != expected_patches:
            if num_patches < expected_patches:
                # Upsample by repeating patches
                repeat_indices = np.tile(np.arange(num_patches), 
                                       (expected_patches // num_patches + 1))[:expected_patches]
                patches = patches[repeat_indices]
            else:
                # Downsample by selecting evenly spaced patches
                indices = np.linspace(0, num_patches - 1, expected_patches, dtype=int)
                patches = patches[indices]
        
        # print(f"   Final patches shape: {patches.shape}")  # Suppressed for API mode
        
        # Compute Mahalanobis distances
        distances = []
        successful_computations = 0
        
        for i in range(self.legacy_H * self.legacy_W):
            try:
                patch = patches[i]
                mu = self.legacy_mus[i]
                cov_inv = self.legacy_cov_invs[i]
                
                # Ensure dimensions match exactly
                if len(patch) != len(mu):
                    # This shouldn't happen now, but handle gracefully
                    if len(patch) < len(mu):
                        patch = np.pad(patch, (0, len(mu) - len(patch)), 'constant')
                    else:
                        patch = patch[:len(mu)]
                
                # Compute Mahalanobis distance with numerical stability
                diff = patch - mu
                try:
                    distance = np.sqrt(np.abs(diff.T @ cov_inv @ diff))
                    distances.append(distance)
                    successful_computations += 1
                except np.linalg.LinAlgError:
                    # Handle singular matrix case
                    distances.append(np.linalg.norm(diff))
                    
            except Exception as e:
                distances.append(0.0)
        
        # print(f"   Successfully computed distances for {successful_computations}/{len(distances)} locations")  # Suppressed for API mode
        
        # Use maximum distance as anomaly score
        if len(distances) > 0:
            anomaly_score = float(np.max(distances))
        else:
            anomaly_score = 0.0
            
        classification = 'anomaly' if anomaly_score > self.threshold else 'normal'

        return anomaly_score, classification

    def predict(self, image_path: str) -> Tuple[float, str]:
        """
        Predict anomaly score and classification for an image.
        
        Args:
            image_path: Path to test image

        Returns:
            Tuple of (anomaly_score, classification)
            - anomaly_score: Maximum Mahalanobis distance across patches
            - classification: 'normal' or 'anomaly' based on threshold
        """
        if hasattr(self, 'use_legacy_format') and self.use_legacy_format:
            return self._predict_legacy(image_path)
        
        # Standard prediction method for new format
        if self.pca is None:
            raise ValueError("Model not trained. Call train() first or load() a trained model.")
        
        # Extract and reduce patch features
        patches = self._extract_patch_features(image_path)
        reduced_patches = self.pca.transform(patches)
        
        # Compute Mahalanobis distances
        distances = np.array([
            mahalanobis(patch, self.mean_vector, self.covariance_inv)
            for patch in reduced_patches
        ])
        
        # Use maximum distance as anomaly score
        anomaly_score = distances.max()
        classification = 'anomaly' if anomaly_score > self.threshold else 'normal'
        
        return anomaly_score, classification


def detect_anomaly(image_path: str, model_path: str) -> dict:
    """
    Convenience function for anomaly detection on a single image.
    
    Args:
        image_path: Path to test image
        model_path: Path to trained PaDiM model
        
    Returns:
        Dictionary containing detection results
    """
    detector = PaDiMDetector()
    detector.load(model_path)
    score, label = detector.predict(image_path)
    
    return {
        'image_path': str(image_path),
        'anomaly_score': score,
        'classification': label,
        'is_anomaly': label == 'anomaly'
    }




Te
