from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *


class DataLoader:

    def __init__(

            self,

            train_df,

            valid_df,

            test_df

    ):

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def create_generators(self):

        train_generator = ImageDataGenerator(

            rescale=1 / 255,

            rotation_range=20,

            zoom_range=0.2,

            horizontal_flip=True

        )

        test_generator = ImageDataGenerator(

            rescale=1 / 255

        )

        train_data = train_generator.flow_from_dataframe(

            self.train_df,

            x_col="Class Path",

            y_col="Class",

            target_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            class_mode=CLASS_MODE

        )

        valid_data = test_generator.flow_from_dataframe(

            self.valid_df,

            x_col="Class Path",

            y_col="Class",

            target_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            class_mode=CLASS_MODE

        )

        test_data = test_generator.flow_from_dataframe(

            self.test_df,

            x_col="Class Path",

            y_col="Class",

            target_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            class_mode=CLASS_MODE,

            shuffle=False

        )

        return train_data, valid_data, test_data