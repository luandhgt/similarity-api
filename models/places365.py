#!/usr/bin/env python3
"""
Places365 ResNet50 Model Utilities
Loads and manages the Places365 pretrained model
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

# Global model instance
_model = None

def load_places365_model():
    """Load Places365 ResNet50 model from downloaded weights"""
    model_path = Path("models/resnet50_places365.pth.tar")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}! Please run: python setup_model.py"
        )
    
    print(f"🔧 Loading Places365 model from {model_path}")
    
    # Create ResNet50 architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 365)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Clean state dict (remove 'module.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        new_state_dict[key] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print("✅ Places365 model loaded successfully")
    return model

def get_places365_model():
    """Get singleton model instance"""
    global _model
    if _model is None:
        _model = load_places365_model()
    return _model