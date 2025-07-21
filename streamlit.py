import streamlit as st
import pandas as pd
import os
import numpy as np
from db import get_cars_by_make
from predict_price import predict_price

# --- Page Configuration ---
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Helper Functions ---
def get_available_makes() -> list:
    """
    Scans the 'models' directory to find available car makes based on saved model files.
    Assumes model files are named like 'make_model.pkl' (e.g., 'BMW_model.pkl').
    """
    models_dir = 'models'
    if not os.path.exists(models_dir):
        st.error(f"'{models_dir}' directory not found. Please ensure models are trained and saved.")
        return []
    
    makes = set()
    for filename in os.listdir(models_dir):
        if filename.endswith("_model.pkl"):
            make = filename.split('_model.pkl')[0]
            makes.add(make)
    
    if not makes:
        st.error(f"No model files found in the '{models_dir}' directory. Please train models first.")
        return ["No models found"] 
    
    return sorted(list(makes))

@st.cache_data # Cache the results to avoid re-running on every widget interaction
def get_features_for_make(make: str) -> dict:
    """
    Loads unique feature values (model, body_type, fuel, gearbox) for a given car make from the database.
    This function is cached to improve performance.
    """
    try:
        data, columns = get_cars_by_make(make)
        df = pd.DataFrame(data, columns=columns)
        
        features = {
            'model': sorted(df['model'].dropna().unique().tolist()) if 'model' in df.columns else [],
            'body_type': sorted(df['body_type'].dropna().unique().tolist()) if 'body_type' in df.columns else [],
            'fuel': sorted(df['fuel'].dropna().unique().tolist()) if 'fuel' in df.columns else [],
            'gearbox': sorted(df['gearbox'].dropna().unique().tolist()) if 'gearbox' in df.columns else []
        }
        return features
    except Exception as e:
        st.error(f"Failed to load features for {make}: {str(e)}")
        return {}
    
def get_default_value(df_col: pd.Series, dtype, fallback):
    """
    Safely gets the mode of a DataFrame column, converting it to the specified dtype.
    If mode cannot be calculated or conversion fails, returns a fallback value.
    """
    try:
        if df_col.empty or df_col.isnull().all():
            return fallback
        
        mode_val = df_col.dropna().mode()[0]
        return dtype(mode_val)
    except (IndexError, TypeError, ValueError):
        return fallback

# --- Streamlit UI ---
st.title("🚗 Car Price Predictor")
st.markdown("Fill in the details below to get a price prediction for your car.")

# --- Sidebar for Inputs ---
st.sidebar.header("Car Parameters")

available_makes = get_available_makes()

if "No models found" in available_makes and len(available_makes) == 1:
    st.sidebar.warning("No car models available for selection.")
    st.stop()

# --- Input Fields ---
selected_make = st.sidebar.selectbox("Car Make", options=available_makes, key="make_select")

features = get_features_for_make(selected_make)

data_for_defaults, columns_for_defaults = get_cars_by_make(selected_make)
df_for_defaults = pd.DataFrame(data_for_defaults, columns=columns_for_defaults)

car_model = st.sidebar.selectbox("Car Model", options=sorted(features.get('model', [])), key="model_select")
body_type = st.sidebar.selectbox("Body Type", options=sorted(features.get('body_type', [])), key="body_type_select")
fuel = st.sidebar.selectbox("Fuel Type", options=sorted(features.get('fuel', [])), key="fuel_select")
gearbox = st.sidebar.selectbox("Gearbox", options=sorted(features.get('gearbox', [])), key="gearbox_select")

default_year = get_default_value(df_for_defaults.get('year', pd.Series(dtype='int')), int, 2018)
default_mileage = get_default_value(df_for_defaults.get('mileage', pd.Series(dtype='int')), int, 50000)
default_engine_volume = get_default_value(df_for_defaults.get('engine_volume', pd.Series(dtype='float')), float, 2.0)
default_engine_power = get_default_value(df_for_defaults.get('engine_power', pd.Series(dtype='int')), int, 150)

# Function to create number input with text fallback and validation
def create_numeric_input(label, default_value, dtype, key_suffix, placeholder_text=None):
    input_value_str = st.sidebar.text_input(
        label,
        value=str(default_value) if default_value is not None else '',
        key=f"{label.lower().replace(' ', '_')}_{key_suffix}",
        placeholder=placeholder_text or f"Enter {label.lower()} (e.g., {default_value})"
    )
    
    parsed_value = None
    if input_value_str:
        try:
            parsed_value = dtype(input_value_str)
        except ValueError:
            st.sidebar.error(f"Please enter a valid number for {label}.")
            return None
    else:
        parsed_value = default_value
    return parsed_value

year = create_numeric_input("Year", default_year, int, "year_input_key")
mileage = create_numeric_input("Mileage (km)", default_mileage, int, "mileage_input_key")
engine_volume = create_numeric_input("Engine Size (L)", default_engine_volume, float, "engine_volume_input_key", f"Enter engine size (e.g., {default_engine_volume:.1f})")
engine_power = create_numeric_input("Engine Power (kW)", default_engine_power, int, "engine_power_input_key")

# --- Prediction Button ---
if st.sidebar.button("Predict Price", use_container_width=True):
    if None in [selected_make, car_model, year, body_type, fuel, gearbox, engine_volume, engine_power, mileage]:
        st.error("Please ensure all mandatory fields are selected/entered correctly.")
    else:
        try:
            user_input_data = {
                'year': year,
                'mileage': mileage,
                'engine_volume': engine_volume,
                'engine_power': engine_power,
                'model': car_model,
                'body_type': body_type,
                'fuel': fuel,
                'gearbox': gearbox
            }

            with st.spinner('Calculating...'):
                predicted_price = predict_price(selected_make, user_input_data)
            
            st.success(f"**Predicted Price: €{predicted_price:,.2f}**")

        except FileNotFoundError as fnfe:
            st.error(f"Error: {fnfe}. Please ensure a model is trained for '{selected_make}'.")
        except ValueError as ve:
            st.error(f"Input data error: {ve}. Please check your inputs.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

# --- Instructions ---
st.info(
    """
    **How to Use:**
    1.  Select the car's **make**, **model**, **body type**, **fuel type**, and **gearbox** from the dropdowns in the sidebar.
    2.  Enter the **year of manufacture**, **mileage**, **engine size**, and **engine power**. Default values are provided based on available data.
    3.  Click the "Predict Price" button.
    """
)
