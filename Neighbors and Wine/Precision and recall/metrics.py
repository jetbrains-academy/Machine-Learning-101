import numpy as np


def precision_recall(y_pred, y_test):
    class_precision_recall = []
    # Here we calculate the precision and recall for each of the unique classes of the
    # testing sample.
    for c in np.unique(y_test):
        # Here we evaluate the number of TP for the class.
        tp =  len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] == c])
        # Here we evaluate the number of FP for the class.
        fp =  len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] != c])
        # Here we evaluate the number of FN for the class.
        fn =  len([i for i in range(len(y_test)) if y_pred[i] != c and y_test[i] == c])
        # Here we calculate the precision for the class.
        precision =  tp / (tp + fp) if tp + fp > 0 else 0.
        # Here we calculate the recall for the class.
        recall =  recall = tp / (tp + fn) if tp + fn > 0 else 0.
        # Here we add a tuple containing the class and its precision and recall to the resulting array.
        class_precision_recall.append((c, precision, recall))
    return class_precision_recall


def print_precision_recall(result):
    for c, precision, recall in result:
        print("class:", c, "\nprecision:", precision, "\nrecall:", recall, "\n")
