import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib
import os
import logging
from db import get_all_car_makes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app_activity.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def evaluate_model(make):
    """
    Evaluates the performance of a trained Linear Regression model for a specific car make.
    It loads the model and its test data, makes predictions, and prints key performance metrics
    along with the most and least accurate predictions.

    Args:
        make (str): The car make for which to evaluate the model (e.g., 'BMW').
    """
    model_path = f'models/{make}_model.pkl'
    logger.info(f"Starting model evaluation for make: {make}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found for make: {make} at {model_path}. Cannot evaluate model.")
        return

    try:
        model_data = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model data from {model_path} for make {make}: {e}", exc_info=True)
        return

    model = model_data['model']
    model_columns = model_data['model_columns']
    X_test = model_data['X_test']
    y_test = model_data['y_test']

    # When loading X_test, it might not contain all possible dummy variables that the model
    # was trained on (e.g., if a specific 'model' or 'fuel type' was not present in the X_test subset).
    # This loop adds any missing columns from 'model_columns' to 'X_test' and fills them with 0.
    # It then reorders X_test to match the training column order.
    # This ensures consistency required by the trained model for prediction.
    for col in model_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[model_columns]

    if X_test.empty or y_test.empty:
        logger.warning(f"Test data (X_test or y_test) is empty for make: {make}. Cannot perform evaluation.")
        return

    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Evaluation Results for model: {make}")
    logger.info(f"R-squared (R²): {score:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.2f} EUR")

    result_df = y_test.to_frame()
    result_df['Predicted'] = y_pred
    result_df['Error'] = (result_df['price'] - result_df['Predicted']).abs()

    top_accurate = result_df.sort_values(by='Error').head(10)
    logger.info("\nTOP 10 Most Accurate Predictions:")
    print(top_accurate[['price', 'Predicted', 'Error']])

    top_inaccurate = result_df.sort_values(by='Error', ascending=False).head(10)
    logger.info("\nTOP 10 Least Accurate Predictions:")
    print(top_inaccurate[['price', 'Predicted', 'Error']])

if __name__ == "__main__":
    all_car_makes = get_all_car_makes()
    if not all_car_makes:
        logger.warning("No car models to evaluate. Run web_scrapper.py and train_and_save_model.py first.")
    else:
        for make in all_car_makes:
            evaluate_model(make)
            logger.info("-" * 50)
    logger.info("evaluate_model.py script finished.")
