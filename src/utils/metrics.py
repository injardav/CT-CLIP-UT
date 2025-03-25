from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from tabulate import tabulate
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math


def calculate_metrics(soft_preds, targets, pathologies):
    """
    Calculate evaluation metrics for a multi-label model.
    
    Args:
        soft_preds (np.ndarray): Softmax/sigmoid probabilities of shape (N, C).
        targets (np.ndarray): Binary ground truth labels of shape (N, C).
        pathologies (List[str]): List of label names.

    Returns:
        dict: Metrics (per-class + macro/micro/mAP/etc.)
    """
    num_classes = len(pathologies)
    hard_preds = np.zeros_like(soft_preds)
    per_class_metrics = {
        "f1": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
    }

    for i, pathology in enumerate(pathologies):
        y_true = targets[:, i]
        y_prob = soft_preds[:, i]

        # ROC + optimal threshold via distance to (0, 1)
        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        dist = np.sqrt((1 - tpr)**2 + fpr**2)
        best_idx = np.argmin(dist)
        best_thresh = thresh[best_idx]

        # Convert soft to hard predictions
        y_pred = (y_prob > best_thresh).astype(int)
        hard_preds[:, i] = y_pred

        # Compute per-class metrics
        per_class_metrics["f1"].append(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        per_class_metrics["precision"].append(precision_score(y_true, y_pred, zero_division=0))
        per_class_metrics["recall"].append(recall_score(y_true, y_pred, zero_division=0))
        per_class_metrics["roc_auc"].append(roc_auc_score(y_true, y_prob))

    # Now compute global/macro/micro metrics
    metrics = {}

    # Hard predictions used for global metrics
    metrics["label_accuracy"] = accuracy_score(targets.flatten(), hard_preds.flatten())

    metrics["per_class_f1"] = per_class_metrics["f1"]
    metrics["macro_f1"] = np.nanmean(per_class_metrics["f1"])
    metrics["micro_f1"] = f1_score(targets, hard_preds, average="micro", zero_division=0)
    metrics["sample_f1"] = f1_score(targets, hard_preds, average="samples", zero_division=0)

    metrics["per_class_precision"] = per_class_metrics["precision"]
    metrics["macro_precision"] = np.nanmean(per_class_metrics["precision"])
    metrics["micro_precision"] = precision_score(targets, hard_preds, average="micro", zero_division=0)

    metrics["per_class_recall"] = per_class_metrics["recall"]
    metrics["macro_recall"] = np.nanmean(per_class_metrics["recall"])
    metrics["micro_recall"] = recall_score(targets, hard_preds, average="micro", zero_division=0)

    metrics["roc_aucs"] = per_class_metrics["roc_auc"]
    metrics["mean_roc_auc"] = np.nanmean(per_class_metrics["roc_auc"])

    # Use soft preds for mAP
    metrics["mAP"] = average_precision_score(targets, soft_preds, average="macro")

    return metrics

def save_metrics(metrics_list, pathologies, results_path):
    """Save calculated metrics for multiple epochs to a text file."""
    with open(results_path / "metrics.txt", "w") as f:
        for epoch, metrics in enumerate(metrics_list):
            f.write(f"Epoch {epoch} Metrics:\n")
            f.write("=" * 40 + "\n")
            f.write(f"Label Accuracy: {metrics['label_accuracy']:.4f}\n")
            f.write(f"Sample F1 Score: {metrics['sample_f1']:.4f}\n")
            f.write(f"Macro F1 Score: {metrics['macro_f1']:.4f}\n")
            f.write(f"Micro F1 Score: {metrics['micro_f1']:.4f}\n")
            f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
            f.write(f"Micro Precision: {metrics['micro_precision']:.4f}\n")
            f.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n")
            f.write(f"Micro Recall: {metrics['micro_recall']:.4f}\n")
            f.write(f"Mean ROC-AUC: {metrics['mean_roc_auc']:.4f}\n")
            f.write(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}\n\n")

            # Collect per-class metrics into a table
            table_data = []
            for i, pathology in enumerate(pathologies):
                # Handle NaNs for ROC-AUC gracefully
                roc_auc_value = metrics['roc_aucs'][i]
                roc_auc_str = f"{roc_auc_value:.4f}" if not np.isnan(roc_auc_value) else "N/A"

                table_data.append([
                    pathology,
                    f"{metrics['per_class_precision'][i]:.4f}",
                    f"{metrics['per_class_recall'][i]:.4f}",
                    f"{metrics['per_class_f1'][i]:.4f}",
                    roc_auc_str
                ])

            # Format the table and write it to the file
            table_str = tabulate(
                table_data,
                headers=["Pathology", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                tablefmt="grid"
            )
            f.write(table_str + "\n\n")

def plot_precision_recall_curve(targets, predictions, pathologies, results_path, epoch=1):
    """Generate and save a single precision-recall curve plot for all pathologies."""
    results_path = results_path / "precision_recall_curves"
    results_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    
    for i, pathology in enumerate(pathologies):
        precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
        pr_auc = average_precision_score(targets[:, i], predictions[:, i]) if len(set(targets[:, i])) > 1 else np.nan
        plt.plot(recall, precision, label=f'{pathology} (AUC={pr_auc:.2f})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / f"epoch_{epoch}_precision_recall_curves.png")
    plt.close()

def plot_roc_curve(targets, predictions, pathologies, results_path, epoch=1):
    """Generate and save a single ROC curve plot for all pathologies."""
    results_path = results_path / "roc_curves"
    results_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    
    for i, pathology in enumerate(pathologies):
        if len(set(targets[:, i])) > 1:  # Ensure at least two classes exist
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            roc_auc = roc_auc_score(targets[:, i], predictions[:, i])
            plt.plot(fpr, tpr, label=f'{pathology} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / f"epoch_{epoch}_roc_curves.png")
    plt.close()

def plot_per_class_f1(metrics, pathologies, results_path, epoch=1):
    """Generate and save a bar chart of per-class F1 scores."""
    results_path = results_path / "f1_scores"
    results_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    
    f1_scores = metrics['per_class_f1']
    bar_colors = ["#1f77b4" if score > 0 else "#d62728" for score in f1_scores]  # Blue for positive, red for zero

    plt.bar(pathologies, f1_scores, color=bar_colors)
    plt.xlabel("Pathology")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)  # Ensure consistent scale
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(results_path / f"epoch_{epoch}_f1_scores.png")
    plt.close()

def plot_all_metrics(metrics_history, results_path):
    """
    Plots all macro/micro metrics over training epochs in a single figure.
    
    Parameters:
    - metrics_history: A list of dictionaries, where each dictionary contains metric values at a specific epoch.
    - results_path: Path to save the output plot.
    """

    metric_names = [
        "label_accuracy", "sample_f1", "macro_f1", "micro_f1", "macro_precision", 
        "micro_precision", "macro_recall", "micro_recall", "mean_roc_auc", "mAP"
    ]

    metric_titles = [
        "Label Accuracy", "Sample F1 Score", "Macro F1 Score", "Micro F1 Score", 
        "Macro Precision", "Micro Precision", "Macro Recall", "Micro Recall", "Macro ROC-AUC", "Mean Average Precision"
    ]

    # Compute dynamic grid size
    n_metrics = len(metric_names)

    if n_metrics == 1:  # Special case: Only 1 plot -> Use 1x1
        n_rows, n_cols = 1, 1
    elif n_metrics == 2:  # Special case: Only 2 plots -> Use 1 row, 2 columns
        n_rows, n_cols = 1, 2
    else:
        n_cols = math.ceil(math.sqrt(n_metrics))  # Try to make it square-like
        n_rows = math.ceil(n_metrics / n_cols)  # Ensure enough rows to fit all plots

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # If there's only 1 plot, axes won't be an array -> Convert to list
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten for easy iteration

    # Collect metric histories
    metric_values = {metric: [] for metric in metric_names}
    for entry in metrics_history:
        for metric in metric_names:
            metric_values[metric].append(entry.get(metric, np.nan))  # Use NaN if metric is missing

    # Plot each metric over epochs
    epochs = np.arange(len(metrics_history))
    for i, metric in enumerate(metric_names):
        axes[i].plot(epochs, metric_values[metric], marker="o", linestyle="-", label=metric_titles[i])
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(metric_titles[i])
        axes[i].set_title(metric_titles[i])
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend()

    # Hide any unused subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])  # Remove extra axes

    plt.suptitle("Training Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(results_path / "all_metrics.png")
    plt.show()
    plt.close()

def plot_training_progress(train_losses, valid_losses, results_path):
    """
    Plots training loss and validation accuracy side-by-side across epochs.

    Args:
        train_losses (Dict) : Dictionary with keys 'steps' and 'epochs', each containing lists of loss values.
        valid_losses (List) : List of averaged validation losses.
        results_path (String) : Path to save the output plot.
    """
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    steps_losses = train_losses['steps']
    epoch_losses = train_losses['epochs']

    # Calculate indices
    step_indices = np.arange(len(steps_losses))
    epoch_indices = np.linspace(0, len(steps_losses) - 1, len(epoch_losses)).astype(int)

    fig, ax = plt.subplots(1, 3, figsize=(21, 9), gridspec_kw={'wspace': 0.3})

    # --- Plot Training Loss (Left) ---
    ax[0].plot(step_indices, steps_losses, color="tab:blue", marker='o', linestyle='-', label='Step Losses')
    ax[0].plot(epoch_indices, epoch_losses, color="tab:green", marker='s', linestyle='--', label='Epoch Losses')

    # Set x-ticks to epoch numbers
    epochs = np.arange(len(epoch_losses))
    ax[0].set_xticks(epoch_indices)
    ax[0].set_xticklabels([str(epoch) for epoch in epochs])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Contrastive Loss")
    ax[0].set_title("Training Loss")
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.5)

    # --- Plot Validation Loss (Right) ---
    ax[1].plot(epochs, valid_losses, color="tab:orange", marker='o', linestyle='-')
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Contrastive Loss")
    ax[1].set_title("Validation Loss")
    ax[1].set_xticks(epochs)
    ax[1].set_xticklabels([str(epoch) for epoch in epochs])
    ax[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Training Progress", fontsize=14, fontweight="bold")
    plt.savefig(results_path / "training_progress.png")
    plt.show()
    plt.close()
