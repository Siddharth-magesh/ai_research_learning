"""
Evaluation module for Siamese Network.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from tqdm import tqdm


class Evaluator:
    """
    Evaluator class for comprehensive model evaluation.
    """
    
    def __init__(self, model, val_loader, config, device):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained Siamese Network model
            val_loader: Validation/test data loader
            config: Configuration object
            device: Device to evaluate on
        """
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.model.eval()
    
    def evaluate(self):
        """
        Perform comprehensive evaluation.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("=" * 60)
        print("EVALUATION")
        print("=" * 60)
        
        # Collect predictions and labels
        all_distances_genuine = []
        all_distances_fake = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                anchor, positive, negative = batch
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Get embeddings
                z_a, z_p, z_n = self.model(anchor, positive, negative, triplet_bool=True)
                
                # Compute distances
                d_ap = F.pairwise_distance(z_a, z_p).cpu().numpy()
                d_an = F.pairwise_distance(z_a, z_n).cpu().numpy()
                
                # Store distances
                all_distances_genuine.extend(d_ap.tolist())
                all_distances_fake.extend(d_an.tolist())
                
                # For binary classification metrics
                # Label: 1 for genuine (same person), 0 for fake (different person)
                all_labels.extend([1] * len(d_ap))  # Genuine pairs
                all_labels.extend([0] * len(d_an))  # Fake pairs
                
                # Predictions based on threshold
                pred_genuine = (d_ap < self.config.threshold_distance).astype(int)
                pred_fake = (d_an >= self.config.threshold_distance).astype(int)
                # For fake pairs, prediction of 1 means correctly identified as fake
                # So we need to invert: if distance >= threshold, it's fake (label 0)
                pred_fake_labels = (d_an < self.config.threshold_distance).astype(int)
                
                all_predictions.extend(pred_genuine.tolist())
                all_predictions.extend(pred_fake_labels.tolist())
        
        # Convert to numpy arrays
        all_distances_genuine = np.array(all_distances_genuine)
        all_distances_fake = np.array(all_distances_fake)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # For ROC curve, we need continuous scores (negative distances for genuine class)
        all_distances = np.concatenate([all_distances_genuine, all_distances_fake])
        # Convert distances to similarity scores (lower distance = higher similarity)
        similarity_scores = -all_distances
        
        try:
            auc = roc_auc_score(all_labels, similarity_scores)
            fpr, tpr, thresholds = roc_curve(all_labels, similarity_scores)
        except:
            auc = 0.0
            fpr, tpr, thresholds = None, None, None
            print("Warning: Could not compute ROC curve")
        
        # Distance statistics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'mean_dist_genuine': np.mean(all_distances_genuine),
            'std_dist_genuine': np.std(all_distances_genuine),
            'mean_dist_fake': np.mean(all_distances_fake),
            'std_dist_fake': np.std(all_distances_fake),
            'distances_genuine': all_distances_genuine,
            'distances_fake': all_distances_fake,
            'roc_curve': (fpr, tpr, thresholds) if fpr is not None else None
        }
        
        # Print metrics
        self.print_metrics(metrics)
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print("\nClassification Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        print("\nDistance Statistics:")
        print(f"  Genuine Pairs: {metrics['mean_dist_genuine']:.4f} ± {metrics['std_dist_genuine']:.4f}")
        print(f"  Fake Pairs:    {metrics['mean_dist_fake']:.4f} ± {metrics['std_dist_fake']:.4f}")
        print(f"  Threshold:     {self.config.threshold_distance:.4f}")
        
        # Calculate separation
        separation = metrics['mean_dist_fake'] - metrics['mean_dist_genuine']
        print(f"  Separation:    {separation:.4f}")
        
        print("=" * 60)
    
    def find_optimal_threshold(self, start=0.1, end=2.0, steps=100):
        """
        Find optimal threshold for classification.
        
        Args:
            start: Start of threshold range
            end: End of threshold range
            steps: Number of thresholds to try
        
        Returns:
            Tuple of (optimal_threshold, best_accuracy)
        """
        print("\n" + "=" * 60)
        print("FINDING OPTIMAL THRESHOLD")
        print("=" * 60)
        
        # Collect all distances and labels
        all_distances_genuine = []
        all_distances_fake = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Computing distances"):
                anchor, positive, negative = batch
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                z_a, z_p, z_n = self.model(anchor, positive, negative, triplet_bool=True)
                
                d_ap = F.pairwise_distance(z_a, z_p).cpu().numpy()
                d_an = F.pairwise_distance(z_a, z_n).cpu().numpy()
                
                all_distances_genuine.extend(d_ap.tolist())
                all_distances_fake.extend(d_an.tolist())
        
        all_distances_genuine = np.array(all_distances_genuine)
        all_distances_fake = np.array(all_distances_fake)
        
        # Try different thresholds
        thresholds = np.linspace(start, end, steps)
        accuracies = []
        
        for threshold in thresholds:
            # Compute accuracy for this threshold
            genuine_correct = np.sum(all_distances_genuine < threshold)
            fake_correct = np.sum(all_distances_fake >= threshold)
            total = len(all_distances_genuine) + len(all_distances_fake)
            accuracy = (genuine_correct + fake_correct) / total
            accuracies.append(accuracy)
        
        # Find best threshold
        best_idx = np.argmax(accuracies)
        optimal_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]
        
        print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Current Threshold: {self.config.threshold_distance:.4f}")
        print(f"Current Accuracy: {accuracies[np.argmin(np.abs(thresholds - self.config.threshold_distance))]:.4f}")
        print("=" * 60)
        
        return optimal_threshold, best_accuracy, thresholds, accuracies
