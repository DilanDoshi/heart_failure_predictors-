# For Helper Functions Related to Metrics and Evaluating Models 

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)


def evaluate_model(y_true, y_pred, y_pred_proba, model_name="Model"):
    """
    Comprehensive evaluation function for binary classification.
    
    Prints:
    1. Accuracy - Overall correctness (misleading with imbalance)
    2. Recall (Sensitivity) - How well we catch deaths (MOST IMPORTANT clinically)
    3. Precision - Of predicted deaths, how many are real
    4. F1 Score - Harmonic mean of precision and recall
    5. ROC-AUC - Best single metric for ranking ability
    6. Confusion Matrix - Breakdown of predictions
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels (0 or 1)
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str
        Name of the model for display
    """
    print("="*70)
    print(f"EVALUATION METRICS: {model_name}")
    print("="*70)
    
    # 1. ACCURACY
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Easy to interpret, but misleading with class imbalance")
    print("A model predicting all 'survived' would get ~73% accuracy!")
    
    # 2. RECALL (SENSITIVITY) - Most important clinically!
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"\nRECALL (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
    print("MOST IMPORTANT METRIC CLINICALLY")
    print("Of all patients who died, what % did we correctly identify?")
    print(f"We caught {recall*100:.1f}% of actual deaths")
    
    # 3. PRECISION
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"\nPRECISION: {precision:.4f} ({precision*100:.2f}%)")
    print("Of all predicted deaths, what % were correct?")
    print(f"{precision*100:.1f}% of predicted deaths were real")
    
    # 4. F1 SCORE
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"\nF1 SCORE: {f1:.4f}")
    print("Harmonic mean of precision and recall")
    print("Good summary for imbalanced classification")
    print(f"Majority baseline: F1 = 0.0 (predicts no deaths)")
    
    # 5. ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print("Measures ranking ability (0.5 = random, 1.0 = perfect)")

    
    # 6. CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nCONFUSION MATRIX:")
    print("="*50)
    print(f"True Negatives (TN):  {tn:4d}  | Correctly predicted survived")
    print(f"False Positives (FP): {fp:4d}  | Incorrectly predicted death")
    print(f"False Negatives (FN): {fn:4d}  | MISSED deaths ")
    print(f"True Positives (TP):  {tp:4d}  | Correctly predicted death")
    print("="*50)
    print(f"Total Predictions: {tn + fp + fn + tp}")
    print(f"Actual Deaths: {fn + tp} | Actual Survived: {tn + fp}")
    
    # Visualization
    print("\nVisual Confusion Matrix:")
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['Survived', 'Died']).plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Accuracy: {accuracy:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}")
    print(f"F1 Score: {f1:.3f} | ROC-AUC: {roc_auc:.3f}")
    print(f"Missed Deaths (FN): {fn} out of {fn+tp} ({fn/(fn+tp)*100:.1f}%)")
    print("="*50 + "\n")
