import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adamax

from tensorflow.keras.applications import Xception

from config import *

from .base_model import BaseModel


class XceptionModel(BaseModel):

    def build(self):
        base_model = Xception(

            include_top=False,

            weights="imagenet",

            input_shape=(*IMAGE_SIZE, CHANNELS)
        )

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output

        x = GlobalAveragePooling2D()(x)

        x = Dropout(0.30)(x)

        x = Dense(128, activation="relu")(x)

        x = Dropout(0.25)(x)

        outputs = Dense(
            NUM_CLASSES,
            activation="softmax"
        )(x)

        model = Model(
            inputs=base_model.input,
            outputs=outputs
        )

        model.compile(

            optimizer=Adamax(
                learning_rate=LEARNING_RATE
            ),

            loss="categorical_crossentropy",

            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]

        )

        self.base_model = base_model
        self.model = model

        return self

    def get_model(self):
        return self.model

    def get_base_model(self):
        return self.base_model