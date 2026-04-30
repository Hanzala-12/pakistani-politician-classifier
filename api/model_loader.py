"""
Model loader for FastAPI application
Loads and caches all trained models
"""

import torch
import pandas as pd
from pathlib import Path
from typing import Dict
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.train import get_model, CLASS_NAMES

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelLoader:
    """Singleton class for loading and caching models"""
    
    _instance = None
    _models: Dict[str, torch.nn.Module] = {}
    _best_model_name: str = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._load_all_models()
        return cls._instance
    
    def _load_all_models(self):
        """Load all trained models from models/saved/"""
        print("Loading models...")
        
        model_dir = Path('models/saved')
        if not model_dir.exists():
            print("Warning: models/saved/ directory not found")
            return
        
        model_files = list(model_dir.glob('*_best.pth'))
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_best', '')
            
            try:
                # Load model
                model = get_model(model_name, num_classes=16)
                checkpoint = torch.load(model_file, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self._models[model_name] = model
                print(f"  ✓ Loaded: {model_name}")
                
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
        
        # Determine best model
        self._determine_best_model()
        
        print(f"\nTotal models loaded: {len(self._models)}")
        if self._best_model_name:
            print(f"Best model: {self._best_model_name}")
    
    def _determine_best_model(self):
        """Determine best model from results/model_comparison.csv"""
        comparison_file = Path('results/model_comparison.csv')
        
        if not comparison_file.exists():
            # Default to first loaded model
            if self._models:
                self._best_model_name = list(self._models.keys())[0]
            return
        
        try:
            df = pd.read_csv(comparison_file)
            
            # Extract accuracy values (remove % sign and convert to float)
            df['Accuracy'] = df['Test Accuracy'].str.rstrip('%').astype(float)
            
            # Get best model
            best_row = df.loc[df['Accuracy'].idxmax()]
            self._best_model_name = best_row['Model']
            
        except Exception as e:
            print(f"Warning: Could not determine best model: {e}")
            if self._models:
                self._best_model_name = list(self._models.keys())[0]
    
    def get_model(self, model_name: str = None) -> torch.nn.Module:
        """
        Get a model by name
        
        Args:
            model_name: Name of the model (None for best model)
        
        Returns:
            Model
        """
        if model_name is None:
            model_name = self._best_model_name
        
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self._models.keys())}")
        
        return self._models[model_name]
    
    def get_available_models(self) -> list:
        """Get list of available model names"""
        return list(self._models.keys())
    
    def get_best_model_name(self) -> str:
        """Get name of best model"""
        return self._best_model_name
    
    @property
    def device(self) -> str:
        """Get device name"""
        return str(device)


# Global instance
model_loader = ModelLoader()
