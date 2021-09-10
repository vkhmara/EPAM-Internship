from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def all_scores(y_true, y_pred, average='macro'):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average)
    }
def all_info(clf, X, y_true):
    predicted = clf.predict(X)
    return {
        average:
            all_scores(y_true, predicted, average=average)
        for average in ['macro', 'micro']
    }