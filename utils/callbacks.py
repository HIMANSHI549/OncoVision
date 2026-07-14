from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


class CallbackFactory:

    @staticmethod
    def get_callbacks():

        early = EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        )

        reduce = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6
        )

        return [early, reduce]