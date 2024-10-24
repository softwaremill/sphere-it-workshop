import yaml
import argparse
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score
from poc_transform.data.reviews import prepare_data, read_data, split
from poc_transform.model.xgb import TfIdfXgbClassifier

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def _load_stop_words(stopwords_path:str) -> list[str]:
    with open(stopwords_path, "r") as stopwords_file:
        stop_words = [line.strip() for line in stopwords_file.readlines()]
    return stop_words


def load_config(config_path:str, stopwords_path:str) -> dict:
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config["vectorizer"]["stop_words"] = _load_stop_words(stopwords_path)
    return config


def log_mlflow_parameter(mlflow, name, parameter):
    if type(parameter) == dict:
        for pname, value in parameter.items():
            log_mlflow_parameter(mlflow, f"{name}.{pname}", value)
    else:
        mlflow.log_param(name, str(parameter)[:500])


def train(config:dict):

    mlflow.set_tracking_uri('http://0.0.0.0:5000')
    experiment = mlflow.set_experiment("sklearn-imdb")

    data = read_data(config["data"]["path"])
    texts, labels = prepare_data(data)
    texts_train, texts_test, labels_train, labels_test = split(texts, labels)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model = TfIdfXgbClassifier(config)
        model.train(texts_train, labels_train)

        predictions = model.predict(texts_test)
        accuracy = accuracy_score(labels_test, predictions)

        log_mlflow_parameter(mlflow, "config", config)

        mlflow.log_metric("Accuracy", accuracy)

        signature = infer_signature(texts_test, labels_test)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgb_tfidf_model",
            signature=signature,
            registered_model_name="xgb-tfidf-model"
        )


def serialize_model(model, output_path):
    with open(output_path, "wb") as out:
        joblib.dump(model, out)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training.yaml")
    parser.add_argument("--stopwords", type=str, default="data/misc/stopwords.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args.stopwords)
    train(config)
