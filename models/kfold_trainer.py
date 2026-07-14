from sklearn.model_selection import StratifiedKFold

from config import *

from data.dataset import DatasetLoader
from models.fold_trainer import FoldTrainer
from models.metrics_tracker import MetricsTracker


class KFoldTrainer:

    def __init__(self):

        self.dataset = DatasetLoader()

        self.df = self.dataset.load_train_dataframe()

        self.tracker = MetricsTracker()

    def run(self):

        skf = StratifiedKFold(

            n_splits=K_FOLDS,

            shuffle=SHUFFLE,

            random_state=RANDOM_STATE

        )

        X = self.df["Class Path"]

        y = self.df["Class"]

        for fold, (train_idx, valid_idx) in enumerate(

                skf.split(X, y),

                start=1

        ):

            print(f"\n{'='*50}")

            print(f"Fold {fold}/{K_FOLDS}")

            print(f"{'='*50}\n")

            train_df = self.df.iloc[train_idx]

            valid_df = self.df.iloc[valid_idx]

            fold_trainer = FoldTrainer(

                train_df,

                valid_df

            )

            train_acc, val_acc, sensitivity, specificity = fold_trainer.train()

            self.tracker.add(

                train_acc,

                val_acc,

                sensitivity,

                specificity

            )

        self.tracker.summary()