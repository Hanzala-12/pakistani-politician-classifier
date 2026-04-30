"""
Evaluation Script for Pakistani Politician Image Classification
Evaluates trained models on test set and generates reports
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from tqdm import tqdm
import mlflow

# Import from train.py
from train import (
    get_model, get_dataloaders, CLASS_NAMES,
    device, class_to_idx, idx_to_class
)


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('val_acc', 0.0)


def evaluate_model(model, test_loader):
    """
    Evaluate model on test set
    
    Returns:
        all_preds, all_labels, all_probs, all_paths
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
    
    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        all_paths
    )


def plot_confusion_matrix(cm, model_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_misclassified(all_preds, all_labels, all_probs, all_paths, model_name, top_k=5):
    """Plot top-k misclassified samples"""
    from PIL import Image
    
    # Find misclassified samples
    misclassified_indices = np.where(all_preds != all_labels)[0]
    
    if len(misclassified_indices) == 0:
        print("  No misclassified samples found!")
        return
    
    # Get confidence scores for predicted class
    confidences = all_probs[misclassified_indices, all_preds[misclassified_indices]]
    
    # Sort by confidence (highest confidence mistakes)
    sorted_indices = np.argsort(confidences)[::-1][:top_k]
    top_misclassified = misclassified_indices[sorted_indices]
    
    # Plot
    fig, axes = plt.subplots(1, min(top_k, len(top_misclassified)), figsize=(20, 4))
    if top_k == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        if idx >= len(top_misclassified):
            ax.axis('off')
            continue
        
        sample_idx = top_misclassified[idx]
        img_path = all_paths[sample_idx]
        true_label = idx_to_class[all_labels[sample_idx]]
        pred_label = idx_to_class[all_preds[sample_idx]]
        confidence = all_probs[sample_idx, all_preds[sample_idx]] * 100
        
        # Load and display image
        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            ax.set_title(
                f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                fontsize=10
            )
            ax.axis('off')
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_top5_misclassified.png', dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_all_models():
    """Evaluate all trained models"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Get test dataloader
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=4)
    
    # Find all trained models
    model_dir = Path('models/saved')
    model_files = list(model_dir.glob('*_best.pth'))
    
    if len(model_files) == 0:
        print("No trained models found in models/saved/")
        return
    
    print(f"\nFound {len(model_files)} trained models")
    
    results_summary = []
    
    for model_file in model_files:
        model_name = model_file.stem.replace('_best', '')
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model = get_model(model_name, num_classes=16)
            model, val_acc = load_checkpoint(model, str(model_file))
            model.eval()
            
            # Evaluate
            all_preds, all_labels, all_probs, all_paths = evaluate_model(model, test_loader)
            
            # 1. Overall Accuracy
            test_acc = accuracy_score(all_labels, all_preds)
            print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")
            
            # 2. Classification Report
            report = classification_report(
                all_labels,
                all_preds,
                target_names=CLASS_NAMES,
                digits=4
            )
            print(f"\nClassification Report:\n{report}")
            
            # Save report
            with open(f'results/{model_name}_classification_report.txt', 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
                f.write(f"Validation Accuracy: {val_acc*100:.2f}%\n\n")
                f.write(report)
            
            # 3. Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(cm, model_name)
            print(f"✓ Confusion matrix saved")
            
            # 4. Top-5 Misclassified
            plot_top_misclassified(all_preds, all_labels, all_probs, all_paths, model_name)
            print(f"✓ Top-5 misclassified samples saved")
            
            # 5. Compute metrics for summary
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels,
                all_preds,
                average='macro',
                zero_division=0
            )
            
            results_summary.append({
                'Model': model_name,
                'Test Accuracy': f'{test_acc*100:.2f}%',
                'Macro Precision': f'{precision:.4f}',
                'Macro Recall': f'{recall:.4f}',
                'Macro F1': f'{f1:.4f}'
            })
            
            # Log to MLflow
            try:
                mlflow.set_experiment("Pakistani-Politician-Classifier")
                with mlflow.start_run(run_name=f"{model_name}_evaluation"):
                    mlflow.log_metrics({
                        "test_accuracy": test_acc,
                        "macro_precision": precision,
                        "macro_recall": recall,
                        "macro_f1": f1
                    })
                    mlflow.log_artifact(f'results/{model_name}_classification_report.txt')
                    mlflow.log_artifact(f'plots/{model_name}_confusion_matrix.png')
                    mlflow.log_artifact(f'plots/{model_name}_top5_misclassified.png')
            except Exception as e:
                print(f"  Warning: MLflow logging failed: {e}")
            
        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {e}")
            continue
    
    # 6. Model Comparison Table
    if results_summary:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        df = pd.DataFrame(results_summary)
        print(f"\n{df.to_string(index=False)}")
        
        # Save to CSV
        df.to_csv('results/model_comparison.csv', index=False)
        print(f"\n✓ Model comparison saved to results/model_comparison.csv")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)


def main():
    """Main evaluation function"""
    evaluate_all_models()


if __name__ == "__main__":
    main()
