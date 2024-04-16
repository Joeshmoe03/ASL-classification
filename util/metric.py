import tensorflow as tf
from tensorflow_addons.metrics import F1Score

def metricFactory(args, num_classes):
    '''
    Choose the metrics based on the command-line arguments.
    '''
    #TODO: EXPAND ON METRICS: https://stackoverflow.com/questions/59353009/list-of-metrics-that-can-be-passed-to-tf-keras-model-compile
    metrics = []
    for metric in args.metric:
        if metric == 'precision':
            metrics.append(tf.keras.metrics.Precision())
        elif metric == 'recall':
            metrics.append(tf.keras.metrics.Recall())
        elif metric == 'f1_score':
            # We use macro average. Rationale: 
            metrics.append(F1Score(num_classes = num_classes, average='macro'))
        elif metric == 'accuracy':
            # Internally tf knows to default to sparse-categorical-accuracy or categorical-accuracy depending on context of loss function and label encoding.
            metrics.append('accuracy')
        else:
            raise ValueError(f"Metric {metric} not supported")
    return metrics
