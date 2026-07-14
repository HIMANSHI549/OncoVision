import random
import numpy as np
import tensorflow as tf

from config import RANDOM_STATE


class Seed:

    @staticmethod
    def set(seed=RANDOM_STATE):

        random.seed(seed)

        np.random.seed(seed)

        tf.random.set_seed(seed)