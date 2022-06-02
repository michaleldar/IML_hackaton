import sklearn.linear_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.ensemble import RakelD

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from preprcessing import preprocessing_train, preprocessing_test


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
            if np.abs(corr) <= 0.0005:
                print(f"Correlation between feature {feature} and label {label} is: {corr}")
            # go.Figure([go.Scatter(x = X[feature], y=y, mode= "markers")])\
            #     .update_layout(
            #     title = f"Price as function of {feature}, {label}, The correlation is: {corr}",
            #     yaxis_title="Price",
            #     xaxis_title= "feature values").show()


if __name__ == '__main__':

    X = pd.read_csv("data/train.feats.csv")
    y = pd.read_csv("data/train.labels.0.csv")

    train_X = X.sample(frac=0.5, random_state=12344)
    train_y = y.sample(frac=0.5, random_state=12344)
    test_X = X.drop(train_X.index)
    test_y = y.drop(train_y.index)

    # X, y = preprocessing_train(pd.read_csv("data/train.feats.csv"), pd.read_csv("data/train.labels.0.csv"))
    # train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.5, random_state=42)
    train_X, train_y = preprocessing_train(train_X, train_y)
    test_X = preprocessing_test(test_X)

    # vizualization_for_features(train_X)
    # feature_evaluation(train_X, train_y)
    rk = RakelD(
        base_classifier = RandomForestClassifier(),
        base_classifier_require_dense=[True, True]
    )
    rk.fit(train_X, train_y)
    pred = rk.predict(test_X)
    y_gold = pd.DataFrame()
    y_pred = pd.DataFrame.sparse.from_spmatrix(pred, columns=train_y.columns).sparse.to_dense()

    y_pred["prediction"] = y_pred.apply(lambda x: str([c for c in x.index if x[c] > 0]), axis=1)
    y_gold["prediction"] = test_y[test_y.columns[0]]
    y_pred["prediction"].to_csv("./y_pred.csv", header=False, index=False)
    y_gold["prediction"].to_csv("./y_gold.csv", header=False, index=False)


    y = pd.read_csv("data/train.labels.1.csv")
    train_X = X.sample(frac=0.5, random_state=42)
    train_y = y.sample(frac=0.5, random_state=42)
    test_X = X.drop(train_X.index)
    test_y = y.drop(train_y.index)

    # X, y = preprocessing_train(pd.read_csv("data/train.feats.csv"), pd.read_csv("data/train.labels.0.csv"))
    # train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.5, random_state=42)
    train_X, train_y = preprocessing_train(train_X, train_y, False)
    test_X = preprocessing_test(test_X)

    # X, y = preprocessing_train(pd.read_csv("../data/train.feats.csv"), pd.read_csv("../data/train.labels.1.csv"), multi_label=False)
    # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=42)
    lr = LinearRegression()
    lr.fit(train_X, train_y)
    pred = lr.predict(test_X)
    print("Std:", test_y["אבחנה-Tumor size"].std())
    print("RMSE:", np.sqrt(sklearn.metrics.mean_squared_error(pred, test_y)))
