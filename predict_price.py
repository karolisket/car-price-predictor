import joblib
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app_activity.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def load_model(make: str):
    """
    Loads a trained model and its associated metadata for a specific car make.

    Args:
        make (str): The car make for which to load the model.

    Returns:
        dict: A dictionary containing the model, model_columns, X_test, y_test,
              categorical_unique_values and original_categorical_cols.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For any other errors during model loading.
    """

    path = f'models/{make}_model.pkl'
    logger.info(f"Attempting to load model for make: {make} from {path}")
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        raise FileNotFoundError(f'Model for {make} not found at {path}')
    try:
        model_data = joblib.load(path)
        logger.info(f"Successfully loaded model for make: {make}.")
        return model_data
    except Exception as e:
        logger.error(f"Error loading model for make {make} from {path}: {e}", exc_info=True)
        raise

def prepare_input(user_input_dict: dict, model_columns: list, original_categorical_cols: list) -> pd.DataFrame:
    """
    Prepares a user's car specifications into a DataFrame format suitable for model prediction.
    This involves creating a DataFrame with all expected model columns and correctly
    handling numerical and one-hot encoded categorical features.

    Args:
        user_input_dict (dict): A dictionary of user-provided car specifications.
        model_columns (list): A list of column names the model was trained on.
        original_categorical_cols (list): A list of original categorical column names.

    Returns:
        pd.DataFrame: A DataFrame with one row, ready for prediction.
    """
    final_input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    for key, value in user_input_dict.items():
        # Handle numerical features (e.g., year, mileage).
        if key in final_input_df.columns:
            final_input_df[key] = value
        # Handle categorical features by identifying and setting the correct dummy variable.
        elif key in original_categorical_cols:
            dummy_col_name = f'{key}_{value}'
            if dummy_col_name in final_input_df.columns:
                final_input_df[dummy_col_name] = 1
            else:
                logger.warning(f"Categorical value '{value}' for column '{key}' not found in model's trained columns. Skipping.")
        else:
            logger.warning(f"User input key '{key}' is not a recognized feature for prediction. Skipping.")
    
    final_input_df = final_input_df[model_columns]

    logger.info(f"Input DataFrame prepared for prediction. Shape: {final_input_df.shape}")
    return final_input_df

def predict_price(make: str, user_input_dict: dict) -> float:
    """
    Predicts the price of a car based on its make and user-provided specifications.

    Args:
        make (str): The car make (e.g., 'BMW').
        user_input_dict (dict): A dictionary of car specifications.

    Returns:
        float: The predicted price rounded to two decimal places.

    Raises:
        FileNotFoundError: If the model for the specified make is not found.
        Exception: For other errors during prediction.
    """
    logger.info(f"Starting price prediction for make: {make} with specs: {user_input_dict}")
    try:
        model_data = load_model(make)
        model = model_data['model']
        model_columns = model_data['model_columns']
        original_categorical_cols = model_data['original_categorical_cols']

        input_df = prepare_input(user_input_dict, model_columns, original_categorical_cols)

        if input_df.empty or input_df.shape[1] != len(model_columns):
            logger.error("Prepared input DataFrame is invalid or has wrong number of columns for prediction.")
            raise ValueError("Invalid input data for prediction.")

        predicted_price = model.predict(input_df)[0]
        logger.info(f"Predicted price for {make} is: {predicted_price:.2f} EUR")
        return round(predicted_price, 2)
    except FileNotFoundError as e:
        logger.error(f"Prediction failed: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction for make {make}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    user_car_specs = {
        'year': 2015,
        'mileage': 208500,
        'engine_volume': 3.0,
        'engine_power': 200,
        'model': 'Q7',
        'fuel': 'Dyzelinas',
        'gearbox': 'Automatinė',
        'body_type': 'Visureigis / Krosoveris'
    }

    predicted = predict_price("Audi", user_car_specs)
    print(f"Prognozuojama Audi kaina: {predicted} EUR")
