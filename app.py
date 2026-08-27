import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# APPLICATION SETTINGS
# ============================================================

DATASET_PATH = "car_data.csv.zip"
CURRENT_YEAR = datetime.now().year

st.set_page_config(
    page_title="Used-Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used-Car Price Prediction")

st.write(
    """
    This machine-learning application estimates a used car's selling
    price using its brand, manufacturing year, kilometers driven,
    fuel type, transmission, seller type, and ownership history.
    """
)


# ============================================================
# LOAD THE DATASET
# ============================================================

@st.cache_data
def load_dataset(file_path):
    """
    Load the dataset.

    This function works when car_data.csv.zip is:
    1. A real ZIP file containing a CSV file, or
    2. A normal CSV file with .zip accidentally added to its name.
    """

    if zipfile.is_zipfile(file_path):

        with zipfile.ZipFile(file_path, "r") as zip_file:

            csv_files = [
                file_name
                for file_name in zip_file.namelist()
                if file_name.lower().endswith(".csv")
            ]

            if not csv_files:
                raise ValueError(
                    "The ZIP file does not contain a CSV file."
                )

            with zip_file.open(csv_files[0]) as csv_file:
                data = pd.read_csv(csv_file)

    else:
        data = pd.read_csv(
            file_path,
            compression=None
        )

    return data


# ============================================================
# CLEAN THE DATASET
# ============================================================

@st.cache_data
def clean_dataset(data):
    """Clean the dataset and create machine-learning features."""

    data = data.copy()

    # Standardize column names.
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Rename common alternative column names.
    alternative_names = {
        "car_name": "name",
        "vehicle_name": "name",
        "model_name": "name",
        "price": "selling_price",
        "sellingprice": "selling_price",
        "selling_price_(inr)": "selling_price",
        "kms_driven": "km_driven",
        "kilometers_driven": "km_driven",
        "kilometres_driven": "km_driven",
        "mileage_driven": "km_driven",
        "seller": "seller_type",
        "seller_type_": "seller_type",
        "transmission_type": "transmission",
        "ownership": "owner",
        "owner_type": "owner"
    }

    data = data.rename(columns=alternative_names)

    required_columns = [
        "name",
        "year",
        "selling_price",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            f"{missing_columns}. "
            f"Available columns are: {data.columns.tolist()}"
        )

    # Remove duplicate records.
    data = data.drop_duplicates().copy()

    # Convert necessary columns to numbers.
    numerical_columns = [
        "year",
        "selling_price",
        "km_driven"
    ]

    for column in numerical_columns:

        if data[column].dtype == "object":
            data[column] = (
                data[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Clean text columns.
    text_columns = [
        "name",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ]

    for column in text_columns:
        data[column] = (
            data[column]
            .astype(str)
            .str.strip()
        )

        data[column] = data[column].replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan
        })

    # Remove rows missing essential information.
    data = data.dropna(
        subset=[
            "name",
            "year",
            "selling_price",
            "km_driven",
            "fuel",
            "seller_type",
            "transmission",
            "owner"
        ]
    )

    # Remove impossible numerical values.
    data = data[
        (data["selling_price"] > 0) &
        (data["km_driven"] >= 0) &
        (data["year"] >= 1980) &
        (data["year"] <= CURRENT_YEAR)
    ].copy()

    # Create car age.
    data["car_age"] = (
        CURRENT_YEAR - data["year"]
    )

    # Extract brand from the first word of the car name.
    data["brand"] = (
        data["name"]
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
    )

    # Keep only the columns used by the application.
    data = data[
        [
            "brand",
            "year",
            "car_age",
            "km_driven",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "selling_price"
        ]
    ].copy()

    if len(data) < 20:
        raise ValueError(
            "The dataset does not contain enough valid records "
            "after cleaning. At least 20 records are required."
        )

    return data


# ============================================================
# TRAIN THE MACHINE-LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(data):
    """Train and evaluate a Random Forest regression model."""

    feature_columns = [
        "brand",
        "car_age",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ]

    target_column = "selling_price"

    X = data[feature_columns]
    y = data[target_column]

    numerical_features = [
        "car_age",
        "km_driven"
    ]

    categorical_features = [
        "brand",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ]

    # Numerical preprocessing.
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    # Categorical preprocessing.
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessing = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    # Random Forest regression pipeline.
    model = Pipeline([
        (
            "preprocessing",
            preprocessing
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Divide data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Train the model.
    model.fit(
        X_train,
        y_train
    )

    # Test predictions.
    test_predictions = model.predict(
        X_test
    )

    # Calculate regression metrics.
    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    r2 = r2_score(
        y_test,
        test_predictions
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    comparison = pd.DataFrame({
        "Actual Price": y_test.reset_index(drop=True),
        "Predicted Price": test_predictions
    })

    comparison["Absolute Error"] = abs(
        comparison["Actual Price"] -
        comparison["Predicted Price"]
    )

    return model, metrics, comparison


# ============================================================
# START THE APPLICATION
# ============================================================

try:

    raw_data = load_dataset(
        DATASET_PATH
    )

    df = clean_dataset(
        raw_data
    )

    model, metrics, comparison = train_model(
        df
    )

except FileNotFoundError:

    st.error(
        f"Dataset file '{DATASET_PATH}' was not found. "
        "Make sure it is in the same folder as app.py."
    )

    st.stop()

except Exception as error:

    st.error(
        f"Application error: {error}"
    )

    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Predict Price",
        "Model Performance",
        "Dataset"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Valid vehicle records: {len(df):,}"
)

st.sidebar.write(
    f"Number of brands: {df['brand'].nunique()}"
)

st.sidebar.write(
    "Model: Random Forest Regressor"
)


# ============================================================
# PRICE-PREDICTION PAGE
# ============================================================

if page == "Predict Price":

    st.header("Predict a Used Car's Price")

    st.write(
        "Enter the car information below."
    )

    left_column, right_column = st.columns(2)

    brands = sorted(
        df["brand"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    fuel_types = sorted(
        df["fuel"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    seller_types = sorted(
        df["seller_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    transmission_types = sorted(
        df["transmission"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    owner_types = sorted(
        df["owner"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    with left_column:

        selected_brand = st.selectbox(
            "Car brand",
            brands
        )

        selected_year = st.number_input(
            "Manufacturing year",
            min_value=1980,
            max_value=CURRENT_YEAR,
            value=max(1980, CURRENT_YEAR - 5),
            step=1
        )

        selected_km = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=2_000_000,
            value=50_000,
            step=1_000
        )

        selected_fuel = st.selectbox(
            "Fuel type",
            fuel_types
        )

    with right_column:

        selected_transmission = st.selectbox(
            "Transmission",
            transmission_types
        )

        selected_seller = st.selectbox(
            "Seller type",
            seller_types
        )

        selected_owner = st.selectbox(
            "Ownership",
            owner_types
        )

    selected_car_age = (
        CURRENT_YEAR - selected_year
    )

    prediction_input = pd.DataFrame([{
        "brand": selected_brand,
        "car_age": selected_car_age,
        "km_driven": selected_km,
        "fuel": selected_fuel,
        "seller_type": selected_seller,
        "transmission": selected_transmission,
        "owner": selected_owner
    }])

    if st.button(
        "Predict Selling Price",
        type="primary",
        use_container_width=True
    ):

        try:

            predicted_price = model.predict(
                prediction_input
            )[0]

            predicted_price = max(
                0,
                predicted_price
            )

            st.success(
                f"Estimated selling price: "
                f"{predicted_price:,.2f}"
            )

            st.subheader(
                "Vehicle Information"
            )

            input_summary = pd.DataFrame({
                "Feature": [
                    "Brand",
                    "Manufacturing year",
                    "Car age",
                    "Kilometers driven",
                    "Fuel type",
                    "Transmission",
                    "Seller type",
                    "Ownership"
                ],
                "Value": [
                    selected_brand,
                    selected_year,
                    selected_car_age,
                    f"{selected_km:,}",
                    selected_fuel,
                    selected_transmission,
                    selected_seller,
                    selected_owner
                ]
            })

            st.dataframe(
                input_summary,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "This is a machine-learning estimate. "
                "It is not an official vehicle valuation."
            )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ============================================================
# MODEL-PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.header(
        "Machine-Learning Model Performance"
    )

    metric_column1, metric_column2, metric_column3 = (
        st.columns(3)
    )

    metric_column1.metric(
        "Mean Absolute Error",
        f"{metrics['MAE']:,.2f}"
    )

    metric_column2.metric(
        "Root Mean Squared Error",
        f"{metrics['RMSE']:,.2f}"
    )

    metric_column3.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )

    st.subheader(
        "Actual Prices Compared with Predicted Prices"
    )

    st.scatter_chart(
        comparison,
        x="Actual Price",
        y="Predicted Price"
    )

    st.subheader(
        "Sample Test Predictions"
    )

    st.dataframe(
        comparison.head(30),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        """
        **MAE** represents the average prediction error.

        **RMSE** gives a larger penalty to large prediction errors.

        **R²** measures how much of the price variation the model
        explains. A value closer to 1 is generally better.
        """
    )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.header(
        "Cleaned Used-Car Dataset"
    )

    information_column1, information_column2 = (
        st.columns(2)
    )

    information_column1.metric(
        "Number of Records",
        f"{len(df):,}"
    )

    information_column2.metric(
        "Number of Brands",
        df["brand"].nunique()
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Numerical Statistics"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader(
        "Cars by Brand"
    )

    brand_counts = (
        df["brand"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(
        brand_counts
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Used-Car Price Prediction | "
    "Python, Pandas, scikit-learn, Random Forest and Streamlit"
)

