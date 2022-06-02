from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from preprcessing import preprocessing


def vizualization_for_features(X: pd.DataFrame):
    pca = PCA(n_components=X.shape[1]).fit(X)
    ev = pca.singular_values_ ** 2
    pd.DataFrame(np.array([ev, ev / sum(ev), pca.explained_variance_ratio_]),
              columns=list(range(1, X.shape[1]+1)),
              index=["Eigenvalues", "Explained Variance", "sklearn's Explained Variance"])


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "."):
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        for label in y.columns:
            cov = np.cov(X[feature], y[label])[0][1]
            std_feature = np.std(X[feature])
            std_y = np.std(y[label])
            corr = cov / std_feature / std_y
            go.Figure([go.Scatter(x = X[feature], y=y, mode= "markers")])\
                .update_layout(
                title = f"Price as function of {feature}, {label}, The correlation is: {corr}",
                yaxis_title="Price",
                xaxis_title= "feature values").show()


if __name__ == '__main__':
    X, y = preprocessing(pd.read_csv("../data/train.feats.csv"), pd.read_csv("../data/train.labels.0.csv"))
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.5, random_state=42)

    # vizualization_for_features(train_X)
    feature_evaluation(X, y[:, 0])
