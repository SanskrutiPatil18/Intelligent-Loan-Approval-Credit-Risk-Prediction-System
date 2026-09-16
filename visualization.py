import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df: pd.DataFrame, title: str = 'Correlation Matrix') -> None:
    """
    Plots a heatmap of the correlation matrix for the given DataFrame.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title(title)
    plt.show()

def plot_feature_importance(feature_names: list, importances: list, title: str = 'Feature Importance') -> None:
    """
    Plots a bar chart of feature importances.
    """
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(title)
    plt.xlabel('Importance Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, title='Receiver Operating Characteristic (ROC) Curve') -> None:
    """
    Plots the ROC curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion_matrix(confusion_matrix_array, labels=['Approved', 'Rejected'], title='Confusion Matrix') -> None:
    """
    Plots a confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_array, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()
