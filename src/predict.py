"""
Prediction Script for Pakistani Politician Image Classification
"""

import torch
import argparse
from PIL import Image
from pathlib import Path

from train import get_model, get_transforms, CLASS_NAMES, device


def load_model(model_name, checkpoint_path):
    """Load trained model"""
    model = get_model(model_name, num_classes=16)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_image(model, image_path, top_k=3):
    """
    Predict politician from image
    
    Args:
        model: Trained model
        image_path: Path to image
        top_k: Number of top predictions to return
    
    Returns:
        List of (class_name, confidence) tuples
    """
    # Load and transform image
    transform = get_transforms('test')
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((CLASS_NAMES[idx], prob.item()))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict Pakistani Politician from Image')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='resnet50', help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
    args = parser.parse_args()
    
    # Default checkpoint path
    if args.checkpoint is None:
        args.checkpoint = f'models/saved/{args.model}_best.pth'
    
    # Check if files exist
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.checkpoint)
    
    # Predict
    print(f"Predicting: {args.image}")
    results = predict_image(model, args.image, top_k=args.top_k)
    
    # Display results
    print(f"\nTop-{args.top_k} Predictions:")
    print("-" * 40)
    for rank, (class_name, confidence) in enumerate(results, 1):
        print(f"{rank}. {class_name:<25} {confidence*100:>6.2f}%")


if __name__ == "__main__":
    main()
