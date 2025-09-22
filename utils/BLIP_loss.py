import torch
import torch.nn as nn
from transformers import BlipForConditionalGeneration, BlipProcessor
from torch.nn.functional import normalize

class BLIPDistanceCalculator:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        """
        BLIP image feature distance calculator (compatible with latest BLIP architecture)
        
        Args:
            model_name: BLIP model name (default: Salesforce/blip-image-captioning-base)
            device: Computation device (auto-selects GPU if available)
        """
        # Auto-select device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load BLIP model and processor (using ConditionalGeneration version)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()  # Set to evaluation mode
        
        # Get model configuration
        self.image_size = self.model.config.vision_config.image_size
        self.hidden_size = self.model.config.vision_config.hidden_size

    def preprocess_image(self, image):
        """
        Preprocess input image with proper dimension handling
        Args:
            image: Input tensor (3, 256, 256) or (bs, 3, 256, 256)
        Returns:
            Resized tensor (bs, 3, 224, 224)
        """
        # Ensure input is in [-1, 1] range
        image = torch.clamp(image, -1.0, 1.0)
        
        # Add batch dimension if missing
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Resize with 4D input handling
        resized_image = torch.nn.functional.interpolate(
            image,
            size=(self.image_size, self.image_size),  # Target (H,W)
            mode='bilinear',
            align_corners=False
        )
        return resized_image

    def get_image_features(self, image):
        """
        Extract BLIP feature vector with dimension handling
        Returns:
            Flattened feature tensor (hidden_size,)
        """
        processed_image = self.preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=processed_image)
            # Get CLS token feature and ensure 1D output
            image_features = vision_outputs[0][:, 0].flatten()  # shape: [hidden_size]
        
        return image_features

    def compute_distance(self, image1, image2, distance_metric='cosine'):
        """
        Compute distance with robust dimension handling
        """
        features1 = self.get_image_features(image1)
        features2 = self.get_image_features(image2)
        
        # Normalize features for stable distance computation
        features1 = features1 / torch.norm(features1)
        features2 = features2 / torch.norm(features2)
        
        if distance_metric == 'cosine':
            return 1 - torch.dot(features1, features2)
        elif distance_metric == 'euclidean':
            return torch.norm(features1 - features2)
        elif distance_metric == 'manhattan':
            return torch.sum(torch.abs(features1 - features2))
        else:
            raise ValueError(f"Unsupported metric: {distance_metric}")

    def __call__(self, image1, image2, distance_metric='cosine'):
        """Simplified calling interface"""
        return self.compute_distance(image1, image2, distance_metric)