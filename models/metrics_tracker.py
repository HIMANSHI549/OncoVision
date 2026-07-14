import numpy as np


class MetricsTracker:

    def __init__(self):

        self.train_accuracy = []
        self.validation_accuracy = []

        self.sensitivity = []
        self.specificity = []

    def add(
        self,
        train_acc,
        val_acc,
        sensitivity,
        specificity
    ):

        self.train_accuracy.append(train_acc)
        self.validation_accuracy.append(val_acc)

        self.sensitivity.append(sensitivity)
        self.specificity.append(specificity)

    def summary(self):

        print("\n========== K-FOLD RESULTS ==========\n")

        print(
            f"Average Train Accuracy : {np.mean(self.train_accuracy)*100:.2f}%"
        )

        print(
            f"Average Validation Accuracy : {np.mean(self.validation_accuracy)*100:.2f}%"
        )

        print(
            f"Average Sensitivity : {np.mean(self.sensitivity)*100:.2f}%"
        )

        print(
            f"Average Specificity : {np.mean(self.specificity)*100:.2f}%"
        )

        print(
            f"Std Accuracy : {np.std(self.validation_accuracy)*100:.2f}%"
        )