import json
import os


class LabelManager:

    @staticmethod
    def save(class_indices, save_dir="saved_models"):

        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(
            save_dir,
            "class_indices.json"
        )

        with open(filepath, "w") as file:

            json.dump(class_indices, file, indent=4)

    @staticmethod
    def load(save_dir="saved_models"):

        filepath = os.path.join(
            save_dir,
            "class_indices.json"
        )

        with open(filepath, "r") as file:

            data = json.load(file)

        return list(data.keys())