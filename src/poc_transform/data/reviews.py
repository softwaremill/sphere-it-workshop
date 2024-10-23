import os
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path:str):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


def split(texts:list, labels:list):
    return train_test_split(texts, labels, test_size=0.2, random_state=42)


def download():
    pass    

def prepare_data(df:pd.DataFrame) -> tuple[list[str], list[int]]:
    texts = df.review.tolist()
    labels = [int(label=='positive') for label in df.sentiment.tolist()]
    return texts, labels

