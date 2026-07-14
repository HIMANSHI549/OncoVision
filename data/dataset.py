from sklearn.model_selection import train_test_split

from utils.helper import DatasetHelper

from config import *


class DatasetLoader:

    """
    Creates train,
    validation,
    and test dataframe.
    """

    def __init__(self):

        self.train_df = None
        self.valid_df = None
        self.test_df = None

    def load(self):

        self.train_df = DatasetHelper.create_dataframe(
            TRAIN_PATH
        )

        test_dataframe = DatasetHelper.create_dataframe(
            TEST_PATH
        )

        self.valid_df, self.test_df = train_test_split(

            test_dataframe,

            train_size=VALIDATION_SPLIT,

            random_state=RANDOM_STATE,

            stratify=test_dataframe["Class"]

        )

        return (
            self.train_df,
            self.valid_df,
            self.test_df
        )

    def load_train_dataframe(self):
        return DatasetHelper.create_dataframe(
            TRAIN_PATH
        )