#!/usr/bin/env python3
"""
Simple API wrapper for Recaptured Image Detection

This provides a clean, simple interface for integrating the recapture detection
into other applications.
"""

from inference import RecaptureDetector
from pathlib import Path
import json

class RecaptureAPI:
    """
    Simple API wrapper for recaptured image detection.
    
    Example usage:
        api = RecaptureAPI()
        result = api.detect_single("path/to/image.jpg")
        print(f"Is recaptured: {result['is_recaptured']}")
        print(f"Confidence: {result['confidence']:.3f}")
    """
    
    def __init__(self, model_path="models/checkpoint_100.pth", device=None):
        """
        Initialize the API.
        
        Args:
            model_path (str): Path to model checkpoint
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.detector = RecaptureDetector(model_path=model_path, device=device)
    
    def detect_single(self, image_path):
        """
        Detect if a single image is recaptured.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Simple result with is_recaptured, confidence, class_name
        """
        result = self.detector.predict_single(image_path, return_probabilities=True)
        return {
            # 'is_recaptured': result['is_recaptured'],
            # 'confidence': result['confidence'],
            # 'class_name': result['class_name'],
            'probabilities': result['probabilities']
        }
    
    def detect_batch(self, folder_path, save_results=True):
        """
        Detect recaptured images in a folder.
        
        Args:
            folder_path (str): Path to folder containing images
            save_results (bool): Whether to save results to JSON file
            
        Returns:
            dict: Summary with counts and detailed results
        """
        output_file = "batch_results.json" if save_results else None
        results = self.detector.predict_batch(
            folder_path, 
            output_file=output_file,
            return_probabilities=True
        )
        
        # Create summary
        successful = [r for r in results if 'error' not in r]
        recaptured_count = sum(1 for r in successful if r.get('is_recaptured', False))
        natural_count = len(successful) - recaptured_count
        
        return {
            'summary': {
                'total_images': len(results),
                'recaptured_images': recaptured_count,
                'natural_images': natural_count,
                'success_rate': len(successful) / len(results) if results else 0
            },
            'results': results
        }
    
    def is_recaptured(self, image_path, threshold=0.5):
        """
        Simple boolean check if image is recaptured.
        
        Args:
            image_path (str): Path to image file
            threshold (float): Confidence threshold (0.5 = 50%)
            
        Returns:
            bool: True if image is likely recaptured
        """
        result = self.detect_single(image_path)
        return result['is_recaptured'] and result['confidence'] >= threshold


# Example usage functions
def quick_check(image_path):
    """Quick function to check if an image is recaptured."""
    api = RecaptureAPI()
    result = api.detect_single(image_path)
    
    print(f"Image: {image_path}")
    print(f"Result: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    return result

def batch_check(folder_path):
    """Quick function to check all images in a folder."""
    api = RecaptureAPI()
    results = api.detect_batch(folder_path)
    
    summary = results['summary']
    print(f"Processed {summary['total_images']} images")
    print(f"Recaptured: {summary['recaptured_images']}")
    print(f"Natural: {summary['natural_images']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    
    return results

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python recapture_api.py image_path.jpg")
        print("  python recapture_api.py folder_path/")
    else:
        path = sys.argv[1]
        if Path(path).is_file():
            quick_check(path)
        elif Path(path).is_dir():
            batch_check(path)
        else:
            print(f"Path not found: {path}") 
