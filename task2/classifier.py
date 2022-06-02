from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from preprcessing import preprocessing


def vizualization_for_features(X: pd.DataFrame):
    pca = PCA(n_components=X.shape[1]).fit(X)
    ev = pca.singular_values_ ** 2
    pd.DataFrame(np.array([ev, ev / sum(ev), pca.explained_variance_ratio_]),
              columns=list(range(1, X.shape[1]+1)),
              index=["Eigenvalues", "Explained Variance", "sklearn's Explained Variance"])



if __name__ == '__main__':
    X, y = preprocessing(pd.read_csv("data/train.feats.csv"), pd.read_csv("data/train.labels.0.csv"))
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.5, random_state=42)

    # vizualization_for_features(train_X)
    preprocessing(train_X)
