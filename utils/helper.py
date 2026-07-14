import os
import pandas as pd


class DatasetHelper:
    """
    Utility class for creating dataframe
    from image folders.
    """

    @staticmethod
    def create_dataframe(dataset_path):

        classes = []
        paths = []

        for label in os.listdir(dataset_path):

            folder = os.path.join(dataset_path, label)

            if os.path.isdir(folder):

                for image in os.listdir(folder):

                    classes.append(label)
                    paths.append(os.path.join(folder, image))

        dataframe = pd.DataFrame({
            "Class Path": paths,
            "Class": classes
        })

        if dataframe.empty:
            raise FileNotFoundError(
                f"No images found in '{dataset_path}'. "
                "Expected subfolders per class with image files inside."
            )

        return dataframe