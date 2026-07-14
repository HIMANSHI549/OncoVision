import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


class ConfusionMatrix:

    @staticmethod
    def plot(y_true, y_pred, labels):

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(7,6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.show()

        return cm