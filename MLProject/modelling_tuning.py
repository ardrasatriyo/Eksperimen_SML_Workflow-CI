import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import sys
import os

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted', zero_division=0)
    recall = recall_score(actual, pred, average='weighted', zero_division=0)
    f1 = f1_score(actual, pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    if not os.path.exists("wine_preprocessing/train.csv"):
        print("Data not found. Pastikan dataset preprocessing tersedia di folder wine_preprocessing.")
        sys.exit(1)
    train_df = pd.read_csv("wine_preprocessing/train.csv")
    test_df = pd.read_csv("wine_preprocessing/test.csv")

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    print("Configuring MLflow tracking URI...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/ardrasatriyo/Eksperimen_SML_Ardra_Chandra_Satriyo.mlflow")
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("Wine_Quality_Tuning_Experiment")
    
    # Enable autolog
    mlflow.sklearn.autolog(disable=True)

    print("Starting hyperparameter tuning...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    with mlflow.start_run(run_name="tuned_rf_model"):
        mlflow.log_params(best_params)

        y_pred = best_model.predict(X_test)
        (accuracy, precision, recall, f1) = eval_metrics(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Best RandomForest: Params={best_params}, Accuracy={accuracy}")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="WineQualityModel_Tuned")
        else:
            mlflow.sklearn.log_model(best_model, "model")
