import os
import requests
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image, UnidentifiedImageError

from tensorflow.keras.models import load_model

from config import *

from utils.labels import LabelManager

class Predictor:

    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model = load_model(model_path)
        self.labels = LabelManager.load()

    def load_image(self, image_path):

        try:

            if image_path.startswith("http"):

                response = requests.get(
                    image_path,
                    timeout=20
                )

                image = Image.open(
                    BytesIO(response.content)
                )

            else:

                image = Image.open(image_path)

            image = image.convert("RGB")

            image = image.resize(IMAGE_SIZE)

            return image

        except UnidentifiedImageError:

            raise Exception("Invalid Image")

    def preprocess(self, image):

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        return image

    def predict(self, image_path):

        image = self.load_image(image_path)

        array = self.preprocess(image)

        prediction = self.model.predict(
            array,
            verbose=0
        )[0]

        index = np.argmax(prediction)

        self.show(image, prediction)

        return {
            "Predicted Class": self.labels[index],
            "Confidence": float(prediction[index])
        }

    def show(self, image, prediction):

        plt.figure(figsize=(12,6))

        plt.subplot(121)

        plt.imshow(image)

        plt.axis("off")

        plt.title("Input Image")

        plt.subplot(122)

        bars = plt.barh(
            self.labels,
            prediction
        )

        for bar in bars:
            width = bar.get_width()

            plt.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center"
            )

        plt.xlim(0,1)

        plt.xlabel("Probability")

        plt.tight_layout()

        plt.show()