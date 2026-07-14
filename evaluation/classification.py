from sklearn.metrics import classification_report


class ClassificationReport:

    @staticmethod
    def generate(y_true, y_pred):

        print("\nClassification Report\n")

        print(classification_report(y_true, y_pred))