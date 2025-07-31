import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_iris_data():
    """Load iris dataset and return as DataFrame"""
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        df["target_name"] = df["target"].map(
            {0: "setosa", 1: "versicolor", 2: "virginica"}
        )

        logger.info(f"Loaded iris dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading iris data: {e}")
        raise


def save_data(df, filepath):
    """Save DataFrame to CSV"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


if __name__ == "__main__":
    # Load data
    df = load_iris_data()

    # Save raw data
    save_data(df, "data/raw/iris.csv")

    # Split data
    X = df.drop(["target", "target_name"], axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create processed datasets
    train_df = X_train.copy()
    train_df["target"] = y_train

    test_df = X_test.copy()
    test_df["target"] = y_test

    # Save processed data
    save_data(train_df, "data/processed/train.csv")
    save_data(test_df, "data/processed/test.csv")

    print("Data preparation completed!")
