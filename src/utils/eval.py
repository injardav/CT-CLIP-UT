import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, matthews_corrcoef, accuracy_score,
    classification_report, f1_score, average_precision_score
)
from sklearn.utils import resample
from tqdm.notebook import tqdm

def compute_mean(stats, is_df=True):
    """
    Compute the mean for specific labels.

    Args:
        stats: DataFrame or dictionary containing statistics.
        is_df: Whether the input is a DataFrame.

    Returns:
        Mean value of the specified labels.
    """
    spec_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    if is_df:
        return np.mean(stats[spec_labels].iloc[0])
    return np.mean([stats[label][0] for label in spec_labels])

def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy for the specified values of k.

    Args:
        output: Model predictions.
        target: Ground truth labels.
        topk: Tuple of top-k values.

    Returns:
        List of accuracies for each top-k value.
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.expand(-1, max(topk)))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def sigmoid(x):
    """
    Apply the sigmoid function.

    Args:
        x: Input value.

    Returns:
        Sigmoid of x.
    """
    return 1 / (1 + np.exp(-x))

def plot_roc(y_pred, y_true, roc_name, plot_dir, plot=True):
    """
    Plot and save the ROC curve.

    Args:
        y_pred: Predicted probabilities.
        y_true: Ground truth labels.
        roc_name: Name of the plot.
        plot_dir: Directory to save the plot.
        plot: Whether to display the plot.

    Returns:
        fpr, tpr, thresholds, roc_auc.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    if plot:
        sns.set_style('white')
        plt.figure(dpi=300)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='#5C5D9E', linewidth=2)
        plt.fill_between(fpr, tpr, color='#5C5D9E', alpha=0.3)
        plt.plot([0, 1], [0, 1], '--', color='#707071', linewidth=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(roc_name)
        plt.legend(loc='lower right')
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.savefig(f"{plot_dir}/{roc_name}.png", bbox_inches='tight')
    return fpr, tpr, thresholds, roc_auc

def choose_operating_point(fpr, tpr, thresholds):
    """
    Choose the optimal operating point based on Youden's J statistic.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        thresholds: Thresholds.

    Returns:
        Sensitivity, specificity.
    """
    optimal_idx = np.argmax(tpr - fpr)
    return tpr[optimal_idx], 1 - fpr[optimal_idx]

def plot_pr(y_pred, y_true, pr_name, plot_dir, plot=True):
    """
    Plot and save the precision-recall curve.

    Args:
        y_pred: Predicted probabilities.
        y_true: Ground truth labels.
        pr_name: Name of the plot.
        plot_dir: Directory to save the plot.
        plot: Whether to display the plot.

    Returns:
        precision, recall, thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    baseline = len(y_true[y_true == 1]) / len(y_true)
    if plot:
        sns.set_style('whitegrid')
        plt.figure(dpi=300)
        plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}', color='#5C5D9E', linewidth=2)
        plt.plot([0, 1], [baseline, baseline], '--', color='#707071', linewidth=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(pr_name)
        plt.legend(loc='lower right')
        plt.savefig(f"{plot_dir}/{pr_name}.jpg", bbox_inches='tight')
    return precision, recall, thresholds

def evaluate(y_pred, y_true, cxr_labels, plot_dir, label_idx_map=None):
    """
    Evaluate the model predictions using ROC and PR curves.

    Args:
        y_pred: Predicted probabilities.
        y_true: Ground truth labels.
        cxr_labels: List of class labels.
        plot_dir: Directory to save plots.
        label_idx_map: Optional mapping of label indices.

    Returns:
        DataFrame containing evaluation metrics.
    """
    num_classes = y_pred.shape[-1]
    dataframes = []
    for i in range(num_classes):
        y_pred_i = y_pred[:, i]
        y_true_i = y_true[:, label_idx_map[cxr_labels[i]]] if label_idx_map else y_true[:, i]
        
        roc_name = f"{cxr_labels[i]} ROC Curve"
        fpr, tpr, thresholds, roc_auc = plot_roc(y_pred_i, y_true_i, roc_name, plot_dir, plot=False)
        df = pd.DataFrame([roc_auc], columns=[f"{cxr_labels[i]}_auc"])
        dataframes.append(df)

        pr_name = f"{cxr_labels[i]} Precision-Recall Curve"
        plot_pr(y_pred_i, y_true_i, pr_name, plot_dir, plot=False)

    return pd.concat(dataframes, axis=1)

def compute_cis(data, confidence_level=0.05):
    """
    Compute confidence intervals from bootstrap samples.

    Args:
        data: DataFrame of bootstrap samples.
        confidence_level: Confidence level for intervals.

    Returns:
        DataFrame with mean, lower, and upper bounds for each label.
    """
    intervals = []
    for col in data.columns:
        sorted_vals = data[col].sort_values()
        lower = sorted_vals.iloc[int(confidence_level / 2 * len(sorted_vals))]
        upper = sorted_vals.iloc[int((1 - confidence_level / 2) * len(sorted_vals))]
        mean = sorted_vals.mean()
        intervals.append(pd.DataFrame({col: [mean, lower, upper]}))
    result = pd.concat(intervals, axis=1)
    result.index = ['mean', 'lower', 'upper']
    return result

def bootstrap(y_pred, y_true, cxr_labels, n_samples=1000, label_idx_map=None):
    """
    Perform bootstrapping to evaluate model performance.

    Args:
        y_pred: Predicted probabilities.
        y_true: Ground truth labels.
        cxr_labels: List of class labels.
        n_samples: Number of bootstrap samples.
        label_idx_map: Optional mapping of label indices.

    Returns:
        Tuple of bootstrap statistics and confidence intervals.
    """
    np.random.seed(97)
    idx = np.arange(len(y_true))
    boot_stats = []
    for _ in tqdm(range(n_samples)):
        sample = resample(idx, replace=True)
        sample_stats = evaluate(y_pred[sample], y_true[sample], cxr_labels, plot_dir=None, label_idx_map=label_idx_map)
        boot_stats.append(sample_stats)
    boot_stats_df = pd.concat(boot_stats)
    return boot_stats_df, compute_cis(boot_stats_df)
