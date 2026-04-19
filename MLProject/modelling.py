import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
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
    if not os.path.exists("data/train.csv"):
        print("Data fallback: generating from scikit-learn for CI...")
        os.makedirs("data", exist_ok=True)
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        data = load_wine(as_frame=True)
        df = data.frame.dropna()
        X = df.drop(columns=['target'])
        y = df['target']
        X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        pd.concat([X_train, y_train], axis=1).to_csv('data/train.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('data/test.csv', index=False)

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    mlflow.sklearn.autolog()

    print("Training basic RandomForest model...")
    with mlflow.start_run(run_name="basic_rf_model"):
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        (accuracy, precision, recall, f1) = eval_metrics(y_test, y_pred)

        print(f"RandomForest: Accuracy={accuracy}")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(rf, "model", registered_model_name="WineQualityModel_Basic")
        else:
            mlflow.sklearn.log_model(rf, "model")
