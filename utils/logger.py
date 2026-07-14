import logging
import os


class Logger:

    @staticmethod
    def setup():

        os.makedirs("logs", exist_ok=True)

        logging.basicConfig(
            filename="logs/training.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        return logging.getLogger("PancreaticCancer")