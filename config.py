"""
Project Configuration File
Author: Himanshi
"""

import os

# =========================
# DATASET PATHS
# =========================

TRAIN_PATH = r"dataset/train/train"
TEST_PATH = r"dataset/test/test"

# =========================
# IMAGE SETTINGS
# =========================

IMAGE_SIZE = (224, 224)

CHANNELS = 3

BATCH_SIZE = 32

NUM_CLASSES = 2
# =========================
# DATA SPLIT
# =========================

VALIDATION_SPLIT = 0.5
# =========================
# TRAINING
# =========================

INITIAL_EPOCHS = 15

FINE_TUNE_EPOCHS = 5

LEARNING_RATE = 1e-4

FINE_TUNE_LR = 1e-5

# =========================
# MODEL
# =========================

MODEL_NAME = "DenseNet"

# Available:
# DenseNet
# EfficientNet
# Xception
# DenseNet_KFold

# =========================
# RANDOM SEED
# =========================

RANDOM_STATE = 42
# =========================
# K-FOLD
# =========================

K_FOLDS = 5

SHUFFLE = True
# =========================
# IMAGE
# =========================

INPUT_SHAPE = (
    IMAGE_SIZE[0],
    IMAGE_SIZE[1],
    CHANNELS
)

CLASS_MODE = "categorical"
# =========================
# SAVE PATH
# =========================
SAVE_DIR = "saved_models"
MODEL_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "pancreatic_model.keras"
)