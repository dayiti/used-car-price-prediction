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


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Used-Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used-Car Price Prediction")
st.write(
    "Enter the vehicle information to estimate its selling price."
)


# ------------------------------------------------------------
# Load CSV from the ZIP file
# ------------------------------------------------------------

@st.cache_data
def load_and_clean_data():
    zip_path = "car_data.csv.zip"

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        csv_files = [
            name for name in zip_file.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                "No CSV file was found inside car_data.csv.zip."
            )

        with zip_file.open(csv_files[0]) as csv_file:
            data = pd.read_csv(csv_file)

    # Clean column names.
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

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
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}. "
            f"Available columns: {data.columns.tolist()}"
        )

    # Remove duplicate records.
    data = data.drop_duplicates().copy()

    # Convert important columns to numbers.
    for column in ["year", "selling_price", "km_driven"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove records containing invalid required values.
    data = data.dropna(
        subset=[
            "name",
            "year",
            "selling_price",
            "km_driven"
        ]
    )

    current_year = datetime.now().year

    data = data[
        (data["selling_price"] > 0) &
        (data["km_driven"] >= 0) &
        (data["year"] >= 1980) &
        (data["year"] <= current_year)
    ].copy()

    # Create useful machine-learning features.
    data["car_age"] = current_year - data["year"]

    data["brand"] = (
        data["name"]
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
    )

    # Keep only the features needed for this project.
    data = data[
        [
            "brand",
            "car_age",
            "km_driven",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "selling_price"
        ]
    ]

    return data


# ------------------------------------------------------------
# Train the machine-learning model
# ------------------------------------------------------------

@st.cache_resource
def train_model(data):
    X = data.drop(columns=["selling_price"])
    y = data["selling_price"]

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

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessing = ColumnTransformer([
        (
            "numbers",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categories",
            categorical_pipeline,
            categorical_features
        )
    ])

    model = Pipeline([
        (
            "preprocessing",
            preprocessing
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(
            y_test,
            predictions
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        ),
        "R2": r2_score(
            y_test,
            predictions
        )
    }

    comparison = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": predictions
    })

    return model, metrics, comparison


# ------------------------------------------------------------
# Load data and model safely
# ------------------------------------------------------------

try:
    df = load_and_clean_data()
    model, metrics, comparison = train_model(df)

except Exception as error:
    st.error(f"Application error: {error}")
    st.stop()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

page = st.sidebar.radio(
    "Choose a page",
    [
        "Predict Price",
        "Model Performance",
        "View Dataset"
    ]
)

st.sidebar.write(f"Valid vehicle records: {len(df):,}")
st.sidebar.write("Model: Random Forest Regression")


# ------------------------------------------------------------
# Price prediction page
# ------------------------------------------------------------

if page == "Predict Price":

    st.header("Enter the Vehicle Information")

    left_column, right_column = st.columns(2)

    with left_column:
        brand = st.selectbox(
            "Brand",
            sorted(df["brand"].unique())
        )

        year = st.number_input(
            "Manufacturing year",
            min_value=1980,
            max_value=datetime.now().year,
            value=datetime.now().year - 5,
            step=1
        )

        km_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            value=50000,
            step=1000
        )

        fuel = st.selectbox(
            "Fuel type",
            sorted(df["fuel"].unique())
        )

    with right_column:
        transmission = st.selectbox(
            "Transmission",
            sorted(df["transmission"].unique())
        )

        seller_type = st.selectbox(
            "Seller type",
            sorted(df["seller_type"].unique())
        )

        owner = st.selectbox(
            "Owner",
            sorted(df["owner"].unique())
        )

    car_age = datetime.now().year - year

    input_data = pd.DataFrame([{
        "brand": brand,
        "car_age": car_age,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }])

    if st.button(
        "Predict Selling Price",
        type="primary",
        use_container_width=True
    ):
        predicted_price = model.predict(input_data)[0]
        predicted_price = max(0, predicted_price)

        st.success(
            f"Estimated selling price: {predicted_price:,.2f}"
        )

        st.info(
            "This is a machine-learning estimate and not an "
            "official vehicle valuation."
        )


# ------------------------------------------------------------
# Model performance page
# ------------------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance")

    column1, column2, column3 = st.columns(3)

    column1.metric(
        "MAE",
        f"{metrics['MAE']:,.2f}"
    )

    column2.metric(
        "RMSE",
        f"{metrics['RMSE']:,.2f}"
    )

    column3.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )

    st.subheader("Sample Test Predictions")

    comparison["Absolute Error"] = abs(
        comparison["Actual Price"] -
        comparison["Predicted Price"]
    )

    st.dataframe(
        comparison.head(30),
        use_container_width=True,
        hide_index=True
    )

    st.scatter_chart(
        comparison,
        x="Actual Price",
        y="Predicted Price"
    )


# ------------------------------------------------------------
# Dataset page
# ------------------------------------------------------------

elif page == "View Dataset":

    st.header("Cleaned Dataset")

    st.write(f"Number of records: {len(df):,}")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "Created with Python, Pandas, scikit-learn and Streamlit"
)

