import numpy as np


class Metrics:

    @staticmethod
    def sensitivity_specificity(cm):

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn + 1e-7)

        specificity = tn / (tn + fp + 1e-7)

        print(f"Sensitivity : {sensitivity:.4f}")

        print(f"Specificity : {specificity:.4f}")

        return sensitivity, specificity