import tensorflow as tf
from tensorflow.keras import backend as K

def f_measure(y_true, y_pred):
    """
    f_measure: Harmonic mean of specificity and sensitivity, shall be used to calculate score batch wise
    during training
    **for binary classification only**
    @param
    y_true: Tensor of actual labels 
    y_pred: Tensor of predicted labels 
    @returns 
    f_measure score for a batch 
    """
    def specificity(y_true, y_pred):
        """Compute the confusion matrix for a set of predictions.
    
        Parameters
        ----------
        y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
        y_true   : correct values for the set of samples used (must be binary: 0 or 1)
    
        Returns
        -------
        out : the specificity
        """
        neg_y_true = 1 - y_true
        neg_y_pred = 1 - y_pred
        fp = K.sum(neg_y_true * y_pred)
        tn = K.sum(neg_y_true * neg_y_pred)
        
        specificity = tn / (tn + fp + K.epsilon())
        return specificity
    
    def recall(y_true, y_pred):
        """Recall metric.
        
        Only computes a batch-wise average of recall.
        
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    specificity = specificity(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((specificity * recall)/(specificity + recall + K.epsilon()))




def f1_score(y_true, y_pred):
    """Computes the F score.
     The F1 score is harmonic mean of precision and recall.
     it is computed as a batch-wise average.
     This is can be used for multi-label classification. 
    """
    
    
    def precision(y_true, y_pred):
        """Precision metric.
         Only computes a batch-wise average of precision.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        """Recall metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_score = 2 * (p * r) / (p + r + K.epsilon())
    return f1_score


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        pass
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass

    def reset_states(self):
        pass