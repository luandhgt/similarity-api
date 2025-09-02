#!/usr/bin/env python3
"""
Setup script to download Places365 ResNet50 model
Run once before starting the API
"""

import os
import requests
from pathlib import Path

def download_places365_model():
    """Download Places365 ResNet50 pretrained weights"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "resnet50_places365.pth.tar"
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        print(f"📊 File size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        return str(model_path)
    
    print("📥 Downloading Places365 ResNet50 model...")
    print("🌐 URL: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar")
    
    url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"📦 Total size: {total_size / 1024 / 1024:.1f} MB")
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📥 Progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", 
                              end="", flush=True)
        
        print(f"\n✅ Downloaded successfully: {model_path}")
        print(f"📊 Final size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        return str(model_path)
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()  # Remove partial file
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if model_path.exists():
            model_path.unlink()
        return None

def verify_model():
    """Verify downloaded model can be loaded"""
    model_path = Path("models/resnet50_places365.pth.tar")
    
    if not model_path.exists():
        print("❌ Model file not found")
        return False
    
    try:
        import torch
        print("🔍 Verifying model file...")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✅ Model format: state_dict")
        else:
            state_dict = checkpoint
            print("✅ Model format: direct")
        
        print(f"📊 Model contains {len(state_dict)} parameters")
        
        # Check for ResNet50 structure
        expected_keys = ['conv1.weight', 'bn1.weight', 'fc.weight', 'fc.bias']
        missing_keys = []
        
        for key in expected_keys:
            if f"module.{key}" not in state_dict and key not in state_dict:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {missing_keys}")
        else:
            print("✅ Model structure verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Places365 ResNet50 model...")
    print("=" * 50)
    
    # Download model
    model_path = download_places365_model()
    
    if model_path:
        print("\n" + "=" * 50)
        # Verify model
        if verify_model():
            print("🎉 Setup completed successfully!")
            print("\n💡 You can now start the API with:")
            print("   uvicorn main:app --reload")
        else:
            print("❌ Setup completed but model verification failed")
    else:
        print("❌ Setup failed - could not download model")
        print("\n🔧 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Verify disk space (need ~100MB)")
        print("   3. Try running again")
        
    print("=" * 50)