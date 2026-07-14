from config import MODEL_NAME

from .densenet_model import DenseNetModel
from .efficientnet_model import EfficientNetModel
from .xception_model import XceptionModel


class ModelFactory:

    @staticmethod
    def get_model():

        if MODEL_NAME == "DenseNet":
            return DenseNetModel()

        elif MODEL_NAME == "EfficientNet":
            return EfficientNetModel()

        elif MODEL_NAME == "Xception":
            return XceptionModel()

        else:
            raise ValueError(f"Unknown model: {MODEL_NAME}")