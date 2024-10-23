from sklearn.metrics import accuracy_score
from poc_transform.data.reviews import prepare_data, read_data, split
from poc_transform.model.xgb import TfIdfXgbClassifier

config = {
    "vectorizer": {
        "max_features": 100,
        "min_df": 0.05,
    },
    "model": {
        "n_jobs": -1
    },
    "data": {
        "path": "data/imdb_part_1.csv",
    },
}

def _load_stop_words():
    with open("data/stopwords.txt", "r") as stopwords_file:
        stop_words = [line.strip() for line in stopwords_file.readlines()]
    config["vectorizer"]["stop_words"] = stop_words

def train():
    _load_stop_words()
    data = read_data(config["data"]["path"])
    texts, labels = prepare_data(data)
    texts_train, texts_test, labels_train, labels_test = split(texts, labels)
    model = TfIdfXgbClassifier(config)

    model.train(texts_train, labels_train)
    predictions = model.test(texts_test, labels_test)
    accuracy = accuracy_score(labels_test, predictions)
    print(f"Accuracy: {accuracy}%")

if __name__ == "__main__":
    train()
