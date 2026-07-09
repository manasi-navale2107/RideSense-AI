import pandas as pd
from src.config import TRAIN_PATH, TEST_PATH


def load_train_data():
    """Load training dataset."""
    return pd.read_csv(TRAIN_PATH)


def load_test_data():
    """Load test dataset."""
    return pd.read_csv(TEST_PATH)


def load_data():
    """Load both train and test datasets."""
    train_df = load_train_data()
    test_df = load_test_data()
    return train_df, test_df