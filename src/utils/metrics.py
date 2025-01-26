from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

def calculate_metrics(predicted_all, real_all, pathologies):
    """Calculate evaluation metrics for the model."""
    metrics = {}
    metrics['flat_accuracy'] = accuracy_score(real_all.flatten(), predicted_all.flatten())
    metrics['per_class_f1'] = f1_score(real_all, predicted_all, average=None)
    metrics['macro_f1'] = f1_score(real_all, predicted_all, average='macro')
    metrics['per_class_precision'] = precision_score(real_all, predicted_all, average=None)
    metrics['per_class_recall'] = recall_score(real_all, predicted_all, average=None)
    metrics['roc_aucs'] = [
        roc_auc_score(real_all[:, i], predicted_all[:, i]) for i in range(len(pathologies))
    ]
    return metrics

def save_metrics(metrics, pathologies, results_path):
    """Save calculated metrics to a text file."""
    with open(results_path / "metrics.txt", "w") as f:
        f.write(f"Flat Accuracy: {metrics['flat_accuracy']:.4f}\n")
        f.write(f"Macro F1 Score: {metrics['macro_f1']:.4f}\n\n")
        for i, pathology in enumerate(pathologies):
            f.write(f"{pathology}:\n")
            f.write(f"  Precision: {metrics['per_class_precision'][i]:.4f}\n")
            f.write(f"  Recall: {metrics['per_class_recall'][i]:.4f}\n")
            f.write(f"  F1 Score: {metrics['per_class_f1'][i]:.4f}\n")
            f.write(f"  ROC-AUC: {metrics['roc_aucs'][i]:.4f}\n\n")

def plot_precision_recall_curve(real_all, predicted_all, pathologies, results_path):
    """Generate and save precision-recall curves for each pathology."""
    for i, pathology in enumerate(pathologies):
        precision, recall, _ = precision_recall_curve(real_all[:, i], predicted_all[:, i])
        plt.figure()
        plt.plot(recall, precision, label=f'PR Curve (AUC={precision.mean():.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for {pathology}")
        plt.legend()
        plt.savefig(results_path / f"PR_Curve_{pathology.replace(' ', '_')}.png")
        plt.close()

def plot_roc_curve(real_all, predicted_all, pathologies, results_path):
    """Generate and save ROC curves for each pathology."""
    for i, pathology in enumerate(pathologies):
        fpr, tpr, _ = roc_curve(real_all[:, i], predicted_all[:, i])
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC={tpr.mean():.2f})')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {pathology}")
        plt.legend()
        plt.savefig(results_path / f"ROC_Curve_{pathology.replace(' ', '_')}.png")
        plt.close()

def plot_per_class_f1(metrics, pathologies, results_path):
    """Generate and save a bar chart of per-class F1 scores."""
    plt.figure()
    plt.bar(pathologies, metrics['per_class_f1'])
    plt.xlabel("Pathology")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(results_path / "Per_Class_F1_Scores.png")
    plt.close()
