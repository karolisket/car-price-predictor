import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import logging
from db import get_cars_by_make, get_all_car_makes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app_activity.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def train_and_save_model(make: str):
    """
    Trains a Linear Regression model for a specific car make using data from the database,
    and saves the model along with necessary metadata to a .pkl file.

    Args:
        make (str): The car make for which to train the model (e.g., 'BMW').
    """
    logger.info(f"Starting model training process for make: {make}")

    data, columns = get_cars_by_make(make)
    df = pd.DataFrame(data, columns=columns)

    df = df.dropna()

    if df.empty:
        logger.warning(f"No sufficient data available for make: {make} after dropping NaNs. Cannot train model.")
        return

    categorical_cols = ['model', 'body_type', 'fuel', 'gearbox']
    
    categorical_unique_values = {}
    for col in categorical_cols:
        if col in df.columns:
            # Collect unique values and sort them for consistent one-hot encoding order during training and prediction.
            unique_vals = sorted(df[col].unique().tolist())
            if unique_vals:
                categorical_unique_values[col] = unique_vals
            else:
                logger.warning(f"Warning: Categorical column '{col}' is empty after NaN removal. Not including in categorical info.")
        else:
            logger.warning(f"Warning: Categorical column '{col}' not found in DataFrame for make: {make}. Check column names.")

    df_encoded = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], dtype=int)
    logger.info(f"DataFrame shape after one-hot encoding: {df_encoded.shape}")

    cols_to_drop_from_X = ['id', 'ad_id', 'make', 'price']
    X_cols = [col for col in df_encoded.columns if col not in cols_to_drop_from_X]

    if 'price' not in df_encoded.columns:
        logger.error(f"'price' column not found in the DataFrame for make: {make}. Cannot train model.")
        return
    
    X = df_encoded[X_cols]
    y = df_encoded['price']

    if X.empty:
        logger.warning(f"Feature DataFrame (X) is empty for make: {make} after encoding/filtering. Cannot train model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Data split into training and testing sets. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info(f"Linear Regression model trained successfully for make: {make}.")

    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created directory for models: {models_dir}")

    # Save the trained model and essential metadata.
    # 'model_columns' is crucial for ensuring prediction inputs match training features.
    # 'X_test', 'y_test' are saved for later model evaluation.
    # 'categorical_unique_values' ensures consistent encoding for new predictions.
    joblib.dump({
        'model': model,
        'model_columns': X_train.columns.tolist(),
        'X_test': X_test,
        'y_test': y_test,
        'categorical_unique_values': categorical_unique_values,
        'original_categorical_cols': categorical_cols
    }, f'{models_dir}/{make}_model.pkl')
    logger.info(f"Model and metadata for make {make} successfully saved to {models_dir}/{make}_model.pkl")

if __name__ == "__main__":
    all_car_makes = get_all_car_makes()
    if not all_car_makes:
        logger.warning("No car makes found in the database. Cannot train any models.")
    else:
        logger.info(f"Found {len(all_car_makes)} car makes to train models for: {', '.join(all_car_makes)}")
        for make in all_car_makes:
            train_and_save_model(make)
            logger.info("-" * 50)
    logger.info("train_and_save_model.py script finished.")
