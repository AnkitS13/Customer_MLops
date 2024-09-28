import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from zenml.client import Client


def load_service():
    """
    Function to load the prediction service from the existing pipeline.
    """
    try:
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=True,
        )
        return service
    except Exception as e:
        st.error(f"Error loading the prediction service: {e}")
        return None


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )

    payment_sequential = st.sidebar.slider("Payment Sequential", 1, 5, 1)
    payment_installments = st.sidebar.slider("Payment Installments", 1, 12, 1)
    payment_value = st.number_input("Payment Value", 0.0, 10000.0, 500.0)
    price = st.number_input("Price", 0.0, 10000.0, 500.0)
    freight_value = st.number_input("Freight Value", 0.0, 1000.0, 50.0)
    product_name_length = st.number_input("Product Name Length", 1, 100, 10)
    product_description_length = st.number_input("Product Description Length", 1, 1000, 100)
    product_photos_qty = st.number_input("Product Photos Quantity", 1, 10, 1)
    product_weight_g = st.number_input("Product Weight (g)", 1, 10000, 500)
    product_length_cm = st.number_input("Product Length (cm)", 1, 100, 10)
    product_height_cm = st.number_input("Product Height (cm)", 1, 100, 10)
    product_width_cm = st.number_input("Product Width (cm)", 1, 100, 10)

    if st.button("Predict"):
        # Load the service
        service = load_service()

        if service is None:
            st.write("No prediction service is currently running.")
            return

        # Prepare the input data for the prediction
        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)

        # Make the prediction using the deployed service
        try:
            service.start(timeout=10)
            pred = service.predict(data)
            st.success(f"Predicted Customer Satisfaction Score: {pred}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()
