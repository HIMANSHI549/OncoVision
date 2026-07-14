import numpy as np

from evaluation.classification import ClassificationReport
from evaluation.confusion import ConfusionMatrix
from evaluation.metrics import Metrics
from evaluation.plots import TrainingPlots


class Evaluator:

    def __init__(self, trainer, test_gen):

        self.trainer = trainer

        self.model = trainer.model

        self.test_gen = test_gen

    def evaluate(self):

        predictions = self.model.predict(
            self.test_gen,
            verbose=1
        )

        y_pred = np.argmax(
            predictions,
            axis=1
        )

        y_true = self.test_gen.classes

        labels = list(
            self.test_gen.class_indices.keys()
        )

        ClassificationReport.generate(
            y_true,
            y_pred
        )

        cm = ConfusionMatrix.plot(
            y_true,
            y_pred,
            labels
        )

        Metrics.sensitivity_specificity(cm)

        history, history_fine = self.trainer.get_history()

        TrainingPlots.plot(
            history,
            history_fine
        )