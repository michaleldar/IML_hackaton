from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == '__main__':
    features = pd.read_csv("../data/train.feats.csv")
    y = pd.read_csv("../data/train.labels.0.csv")
    train_X, train_y, test_x, test_y = train_test_split(features, y, stratify=y)
