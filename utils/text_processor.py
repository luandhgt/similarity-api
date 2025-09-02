#!/usr/bin/env python3
"""
Text processing utilities for Voyage-3-Large embedding
"""

import os
import requests
import numpy as np
from typing import List, Union
import re


class VoyageClient:
    """Client for Voyage-3-Large API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")
        
        self.base_url = "https://api.voyageai.com/v1/embeddings"
        self.model = "voyage-3-large"
        
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Extract embedding vector(s) from text using Voyage-3-Large
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            numpy.ndarray: Embedding vector(s)
                - Single text: shape (1024,)
                - Multiple texts: shape (n_texts, 1024)
        """
        
        # Ensure text is a list
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts,
            "model": self.model
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item['embedding'] for item in result['data']]
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            # Return single vector for single input
            if single_input:
                return embeddings_array.squeeze()
            
            return embeddings_array
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Voyage API request failed: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected API response format: {e}")


def preprocess_text(text: str) -> str:
    """
    Preprocess text before embedding
    
    Args:
        text: Raw text input
        
    Returns:
        str: Preprocessed text
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Basic preprocessing
    text = text.strip()
    
    # Remove excessive whitespace
    text = re.sub(r'#+\s*', '', text)  # Remove headers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text


def extract_text_features(text: str, voyage_client: VoyageClient = None) -> np.ndarray:
    """
    Extract 1024-dimensional feature vector from text
    
    Args:
        text: Input text
        voyage_client: Optional VoyageClient instance (will create new if None)
        
    Returns:
        numpy.ndarray: 1024-dimensional feature vector
    """
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Create client if not provided
    if voyage_client is None:
        voyage_client = VoyageClient()
    
    # Extract embedding
    vector = voyage_client.embed_text(processed_text)
    
    return vector


def validate_text_input(text: str) -> bool:
    """
    Validate text input
    
    Args:
        text: Input text
        
    Returns:
        bool: True if valid
        
    Raises:
        Exception: If validation fails
    """
    if not text:
        raise Exception("Text cannot be None")
    
    if not text.strip():
        raise Exception("Text cannot be empty or only whitespace")
    
    # Check reasonable length (adjust as needed)
    if len(text.strip()) < 3:
        raise Exception("Text too short (minimum 3 characters)")
    
    if len(text) > 100000:  # 100k chars limit
        raise Exception("Text too long (maximum 100,000 characters)")
    
    return True


# Global client instance (lazy loading)
_voyage_client = None

def get_voyage_client() -> VoyageClient:
    """Get singleton VoyageClient instance"""
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = VoyageClient()
    return _voyage_client