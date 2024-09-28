from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    train_pipeline(ingest_data,clean_data,train_model,evaluation)#data_path='C:/Users/Ankit Singh/OneDrive/Desktop/MLops/data/olist_customers_dataset.csv')
    training = train_pipeline(
        ingest_data=ingest_data,    # Pass function names, not executed functions
        clean_data=clean_data,
        model_train=train_model,
        evaluation=evaluation
    )

    training.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )