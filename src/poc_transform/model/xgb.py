import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfXgbClassifier:

    def __init__(self, config:dict):
        self.vectorizer = TfidfVectorizer(**config["vectorizer"])
        self.model = xgb.XGBClassifier(**config["model"])

    def train(self, X, y):
        X_vectorized = self.vectorizer.fit_transform(X)
        self.model.fit(X_vectorized, y)

    def predict(self, X):
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict(X_vectorized)


