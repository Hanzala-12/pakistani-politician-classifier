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
# Avoid importing heavy training module at import-time (mlflow, timm).
# `get_model` will be imported lazily only when needed for classifier checkpoints.

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

        # search common model directories (project_outputs preferred)
        model_dirs = [
            Path('project_outputs/models'),
            Path('models/saved'),
            Path('models'),
            Path('project_outputs/models/saved')
        ]

        model_files = []
        for d in model_dirs:
            if d.exists():
                model_files.extend(list(d.glob('*_best.pth')))

        if not model_files:
            print("Warning: no model checkpoints found in expected locations:")
            for d in model_dirs:
                print(f"  - {d}")
            return
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_best', '')

            try:
                # Inspect checkpoint only (defer heavy model instantiation until requested)
                checkpoint = torch.load(model_file, map_location='cpu')
                keys = list(checkpoint.keys())
                # store path + summary info; actual object created on demand
                self._models[model_name] = {
                    'path': model_file,
                    'keys': keys,
                    'class_names': checkpoint.get('class_names')
                }
                print(f"  [OK] Registered: {model_name} (checkpoint keys: {keys})")

            except Exception as e:
                print(f"  Failed to register {model_name}: {e}")
        
        # Determine best model
        self._determine_best_model()
        
        print(f"\nTotal models loaded: {len(self._models)}")
        if self._best_model_name:
            print(f"Best model: {self._best_model_name}")
    
    def _determine_best_model(self):
        """Determine best model from results/model_comparison.csv"""
        # Prefer project_outputs results, fall back to results/
        candidates = [Path('project_outputs/results/model_comparison.csv'), Path('results/model_comparison.csv')]

        comparison_file = next((p for p in candidates if p.exists()), None)

        if comparison_file is None:
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

        stored = self._models[model_name]

        # If we stored a path/summary, instantiate a runtime predictor via backend.model_loader
        if isinstance(stored, dict) and 'path' in stored:
            try:
                from backend.model_loader import get_predictor
                predictor = get_predictor(model_key=model_name)
                return predictor
            except Exception:
                # Fallback: return stored metadata if backend loader is unavailable
                return stored

        return stored
    
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
