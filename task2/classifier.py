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
    sigma_y = np.std(y)
    features_corr = []
    for feature in X.columns:
        col = X.loc[:, feature]
        features_corr.append(np.cov(col.T, y.T)[0][1] / (np.std(col) * sigma_y))
        go.Figure([go.Scatter(x=X.loc[:, feature], y=y, mode='markers',
                              name=r'$\widehat\mu$')],
                  layout=go.Layout(
                      title="{feature_name} effect on price. Condition's correlation is {correlation}".format(feature_name=feature, correlation=features_corr[-1]),
                      xaxis_title=r"$\text{feature}$",
                      yaxis_title="Price",
                      height=300)).write_image(output_path + "/condition_to_price.png")

    fig = go.Figure([go.Bar(x=X.columns, y=features_corr,
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Features correlations of house price}$",
                  xaxis_title="$\text{ number of samples}$",
                  yaxis_title="Pearson Correlation",
                  height=300))
    fig.show()


if __name__ == '__main__':
    X, y = preprocessing(pd.read_csv("../data/train.feats.csv"), pd.read_csv("../data/train.labels.0.csv"))
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.5, random_state=42)

    # vizualization_for_features(train_X)
    feature_evaluation(X, y[:, 0])
