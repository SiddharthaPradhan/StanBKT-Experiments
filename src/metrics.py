from sklearn.metrics import root_mean_squared_error, roc_auc_score, accuracy_score

def compute_metrics(true_values, pred_values):
    metrics = {}
    metrics['accuracy'] = accuracy_score(true_values, pred_values >= 0.5)
    metrics['auc'] = roc_auc_score(true_values, pred_values)
    metrics['rmse'] = root_mean_squared_error(true_values, pred_values)
    return metrics