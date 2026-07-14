from utils.seed import Seed
from utils.gpu import GPU
from utils.logger import Logger
import os

GPU.disable()

Seed.set()

logger = Logger.setup()

logger.info("Training Started")
from data.dataset import DatasetLoader
from data.dataloader import DataLoader

from models.model_factory import ModelFactory
from models.trainer import Trainer


def main():

    # ==========================
    # LOAD DATASET
    # ==========================
    dataset = DatasetLoader()

    train_df, valid_df, test_df = dataset.load()

    # ==========================
    # CREATE DATA GENERATORS
    # ==========================
    loader = DataLoader(
        train_df,
        valid_df,
        test_df
    )

    train_gen, valid_gen, test_gen = loader.create_generators()

    # ==========================
    # BUILD MODEL
    # ==========================
    model_object = ModelFactory.get_model().build()

    # Optional: Display model summary
    model_object.get_model().summary()

    # ==========================
    # TRAIN MODEL
    # ==========================
    trainer = Trainer(model_object)

    trainer.train(
        train_gen,
        valid_gen
    )

    # ==========================
    # FINE-TUNE MODEL
    # ==========================
    trainer.fine_tune(
        train_gen,
        valid_gen
    )

    # ==========================
    # EVALUATE MODEL
    # ==========================
    trainer.evaluate(
        train_gen,
        valid_gen,
        test_gen
    )

    # ==========================
    # SAVE MODEL
    # ==========================
    trainer.save(train_gen)

    sample_image = r"dataset/test/test/pancreatic_tumor/1-002.jpg"
    if os.path.exists(sample_image):
        from prediction.predictor import Predictor

        predictor = Predictor()

        print(predictor.predict(sample_image))
    else:
        print(f"Skipping sample prediction; image not found: {sample_image}")

    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        trainer,
        test_gen
    )

    evaluator.evaluate()
    print("\n Training Completed Successfully!")
    logger.info("Training Finished Successfully")


if __name__ == "__main__":
    main()