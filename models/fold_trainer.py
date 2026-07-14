import numpy as np

from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *
from models.model_factory import ModelFactory
from models.trainer import Trainer
from data.dataloader import DataLoader

class FoldTrainer:

    def __init__(self, train_df, valid_df):

        self.train_df = train_df
        self.valid_df = valid_df

    def create_generators(self):
        loader = DataLoader(

            self.train_df,

            self.valid_df,

            self.valid_df

        )

        train_gen, valid_gen, _ = loader.create_generators()

        return train_gen, valid_gen
    def train(self):

        train_gen, valid_gen = self.create_generators()

        model_object = ModelFactory.get_model().build()

        trainer = Trainer(model_object)

        trainer.train(
            train_gen,
            valid_gen
        )

        trainer.fine_tune(
            train_gen,
            valid_gen
        )

        train_score = trainer.model.evaluate(
            train_gen,
            verbose=0
        )

        valid_score = trainer.model.evaluate(
            valid_gen,
            verbose=0
        )

        predictions = trainer.model.predict(
            valid_gen,
            verbose=0
        )

        y_pred = np.argmax(
            predictions,
            axis=1
        )

        y_true = valid_gen.classes

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn + 1e-7)

        specificity = tn / (tn + fp + 1e-7)

        return (
            train_score[1],
            valid_score[1],
            sensitivity,
            specificity
        )