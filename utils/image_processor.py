#!/usr/bin/env python3
"""
Image processing utilities for Places365 feature extraction
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from models.places365 import get_places365_model


def preprocess_image(image_path):
    """
    Preprocess image: grayscale -> maintain aspect ratio -> pad to 224x224
    Same preprocessing as original Places365 code
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale then back to RGB (removes color bias)
        img = ImageOps.grayscale(img).convert('RGB')
        
        # Calculate resize dimensions to maintain aspect ratio
        w, h = img.size
        target_size = 224
        
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        # Resize maintaining aspect ratio
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create black canvas 224x224
        canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        
        # Calculate position to center the image
        x = (target_size - new_w) // 2
        y = (target_size - new_h) // 2
        
        # Paste resized image onto black canvas
        canvas.paste(img, (x, y))
        
        return canvas
        
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {e}")


def create_image_transform():
    """Create transformation for Places365 model"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def extract_image_features(image_path):
    """
    Extract 2048-dimensional feature vector from image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy.ndarray: 2048-dimensional feature vector
    """
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Transform for model
    transform = create_image_transform()
    img_tensor = transform(processed_img).unsqueeze(0)
    
    # Get model
    model = get_places365_model()
    
    # Extract features (before final classification layer)
    with torch.no_grad():
        # Get features from the layer before fc
        features = model.avgpool(model.layer4(
            model.layer3(model.layer2(model.layer1(
                model.maxpool(model.relu(model.bn1(model.conv1(img_tensor))))
            )))
        ))
        features = torch.flatten(features, 1)  # Shape: [1, 2048]
        vector = features.cpu().numpy().squeeze()  # Shape: [2048]
    
    return vector


def validate_image_file(image_path):
    """
    Validate if file exists and is a valid image
    Case-insensitive file extension matching for cross-platform compatibility

    Args:
        image_path: Path to image file

    Returns:
        str: Actual path to the file (with correct case)

    Raises:
        Exception: If validation fails
    """
    import os
    import itertools

    def generate_case_variations(text):
        """Generate all possible case variations of a string"""
        if not text:
            return [text]

        # Generate all combinations of upper/lower for each character
        variations = set()
        for combo in itertools.product(*[(c.lower(), c.upper()) for c in text]):
            variations.add(''.join(combo))
        return list(variations)

    # Check if exact path exists
    if os.path.exists(image_path):
        actual_path = image_path
    else:
        # Try case-insensitive matching
        directory = os.path.dirname(image_path)
        filename_with_ext = os.path.basename(image_path)

        # Split into name and extension
        name_parts = filename_with_ext.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, original_ext = name_parts

            # Common image extensions to try
            extensions_to_try = [original_ext, 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']

            actual_path = None
            for ext in extensions_to_try:
                # Generate all case variations for this extension
                ext_variations = generate_case_variations(ext)

                for ext_var in ext_variations:
                    candidate_path = os.path.join(directory, f"{base_name}.{ext_var}")
                    if os.path.exists(candidate_path):
                        actual_path = candidate_path
                        break

                if actual_path:
                    break

            if actual_path is None:
                raise Exception(f"Image file not found: {image_path} (tried case variations and common extensions)")
        else:
            raise Exception(f"Image file not found: {image_path}")

    # Check if file is valid image
    try:
        with Image.open(actual_path) as img:
            pass  # Just check if it opens
        return actual_path
    except Exception as e:
        raise Exception(f"Invalid image file: {actual_path} - {e}")