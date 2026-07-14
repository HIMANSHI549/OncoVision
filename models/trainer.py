import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from config import *

from utils.callbacks import CallbackFactory
from utils.labels import LabelManager

class Trainer:

    def __init__(self, model_object):

        self.model_object = model_object

        self.model = model_object.get_model()

        self.base_model = model_object.get_base_model()

        self.history = None

        self.history_fine = None

    def train(self, train_gen, valid_gen):

        self.history = self.model.fit(

            train_gen,

            validation_data=valid_gen,

            epochs=INITIAL_EPOCHS,

            callbacks=CallbackFactory.get_callbacks()

        )

    def fine_tune(self, train_gen, valid_gen):

        for layer in self.base_model.layers[-40:]:

            layer.trainable = True

        self.model.compile(

            optimizer=Adam(
                learning_rate=FINE_TUNE_LR
            ),

            loss="categorical_crossentropy",

            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]

        )

        self.history_fine = self.model.fit(

            train_gen,

            validation_data=valid_gen,

            epochs=FINE_TUNE_EPOCHS,

            callbacks=CallbackFactory.get_callbacks()

        )

    def evaluate(self, train_gen, valid_gen, test_gen):

        train_score = self.model.evaluate(train_gen)

        valid_score = self.model.evaluate(valid_gen)

        test_score = self.model.evaluate(test_gen)

        print(f"\nTrain Accuracy : {train_score[1]*100:.2f}%")

        print(f"Validation Accuracy : {valid_score[1]*100:.2f}%")

        print(f"Test Accuracy : {test_score[1]*100:.2f}%")

        return train_score, valid_score, test_score

    def save(self, train_gen):
        self.model.save(MODEL_SAVE_PATH)

        LabelManager.save(
            train_gen.class_indices
        )

        print("\nModel Saved Successfully")
    def get_history(self):

        return self.history, self.history_fine