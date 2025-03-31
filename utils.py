from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def evaluate_model(y_true, y_pred, average='weighted'):
    """
    Compute and return classification evaluation metrics.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        average (str): Averaging method for multi-class metrics. Default is 'weighted'.
        
    Returns:
        dict: Dictionary containing accuracy, f1-score, precision, recall, and roc-auc-score.
    """


    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average=average),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average),
        # 'roc_auc': roc_auc_score(y_true, y_pred, multi_class='ovr') if len(set(y_true)) > 2 else roc_auc_score(y_true, y_pred)
        'roc_auc': roc_auc_score(y_true, y_pred)
    }

    print('Model performance')
    print(f"- Accuracy: {metrics['accuracy']}")
    print(f"- F1 Score: {metrics['f1_score']}")
    print(f"- Precision Score: {metrics['precision']}")
    print(f"- Recall Score: {metrics['accuracy']}")
    print(f"- Roc Auc Score: {metrics['roc_auc']}")

    return metrics
