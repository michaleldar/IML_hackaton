""" Usage:
    <file-name> --train_feat=GOLD_FILE --test_feat=PRED_FILE --labels0=LABELS0 --labels1=LABELS1[--debug]

Options:
  --help                           Show this message and exit
  -i INPUT_FILE --in=INPUT_FILE    Input file
                                   [default: infile.tmp]
  -o INPUT_FILE --out=OUTPUT_FILE  Input file
                                   [default: outfile.tmp]
  --debug                          Whether to debug
"""
from pathlib import Path

import sklearn.linear_model
from docopt import docopt
from plotly import subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor, BaggingClassifier
from skmultilearn.ensemble import RakelD
from sklearn.feature_selection import RFE

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
    variance = list(np.around(100 * pca.explained_variance_ratio_, 2)) + [100]

    fig = subplots.make_subplots(rows=1, cols=2,
                        subplot_titles=[r"$\text{Eigenvalues}$", r"$\text{Cumulative Explained Variance}$"],
                        specs=[[{'type': 'Bar'}, {'type': 'Waterfall'}]])

    fig.add_traces([go.Bar(x=['PC1', 'PC2', 'PC3'], y=pca.singular_values_),
                    go.Waterfall(x=["PC1", "PC2", "PC3", "Total"],
                                 y=variance,
                                 text=[f"{v}%" for v in variance],
                                 textposition="outside",
                                 totals={"marker": {"color": "black"}},
                                 measure=["relative", "relative", "relative", "total"])],
                   rows=[1, 1], cols=[1, 2])

    fig.add_shape(type="rect", xref="x", yref="y", x0=-0.4, x1=0.4, y0=0.0, y1=fig.data[1].y[0],
                  line=dict(color="gray"), opacity=1, row=1, col=2)
    fig.add_shape(type="rect", xref="x", yref="y", x0=0.6, x1=1.4, y0=fig.data[1].y[0],
                  y1=fig.data[1].y[0] + fig.data[1].y[1],
                  line=dict(color="pink"), opacity=1, row=1, col=2)
    fig.add_shape(type="rect", xref="x", yref="y", x0=1.6, x1=2.4, y0=fig.data[1].y[0] + fig.data[1].y[1],
                  y1=fig.data[1].y[0] + fig.data[1].y[1] + fig.data[1].y[2],
                  fillcolor="black", line=dict(color="black"), opacity=1, row=1, col=2)

    fig.update_layout(showlegend=False, title=r"$\text{(4) PCA Explained Variance}$", margin=dict(t=100))
    fig.show()


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

def model_selection():
    X = pd.read_csv("data/train.feats.csv")
    y = pd.read_csv("data/train.labels.0.csv")

    ids = X["id-hushed_internalpatientid"].unique()
    train_inds = np.random.choice(ids.shape[0], ids.shape[0] // 2, replace=False)
    train_ids = ids[train_inds]
    train_X = X[X["id-hushed_internalpatientid"].isin(train_ids)]
    train_y = y.loc[train_X.index]
    test_X = X.drop(train_X.index)
    test_y = y.drop(train_y.index)

    # X, y = preprocessing_train(pd.read_csv("data/train.feats.csv"), pd.read_csv("data/train.labels.0.csv"))
    # train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.5, random_state=42)
    train_X, train_y = preprocessing_train(train_X, train_y)
    test_X = preprocessing_test(test_X)

    # vizualization_for_features(train_X)
    # feature_evaluation(train_X, train_y)
    # rk = RakelD(
    #     base_classifier = RandomForestClassifier(),
    #     base_classifier_require_dense=[True, True]
    # )
    # rk.fit(train_X, train_y)
    # pred = rk.predict(test_X)
    rfc = RandomForestClassifier()
    bgc = BaggingClassifier()
    y_gold = pd.DataFrame()
    # y_pred = pd.DataFrame.sparse.from_spmatrix(pred, columns=train_y.columns).sparse.to_dense()
    y_pred = pd.DataFrame()
    for column in train_y.columns:
        rfc.fit(train_X, train_y[column])
        bgc.fit(train_X, train_y[column])
        preds = [rfc.predict(test_X), bgc.predict(test_X)]
        y_pred[column] = np.max(preds, axis=0)

    y_pred["prediction"] = y_pred.apply(lambda x: str([c for c in x.index if x[c] > 0]), axis=1)
    y_gold["prediction"] = test_y[test_y.columns[0]]
    y_pred["prediction"].to_csv("./y_pred.csv", header=False, index=False)
    y_gold["prediction"].to_csv("./y_gold.csv", header=False, index=False)


    y = pd.read_csv("data/train.labels.1.csv")
    ids = X["id-hushed_internalpatientid"].unique()
    train_inds = np.random.choice(ids.shape[0], ids.shape[0] // 2, replace=False)
    train_ids = ids[train_inds]
    train_X = X[X["id-hushed_internalpatientid"].isin(train_ids)]
    train_y = y.loc[train_X.index]
    test_X = X.drop(train_X.index)
    test_y = y.drop(train_y.index)

    # X, y = preprocessing_train(pd.read_csv("data/train.feats.csv"), pd.read_csv("data/train.labels.0.csv"))
    # train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.5, random_state=42)
    train_X, train_y = preprocessing_train(train_X, train_y, False)
    test_X = preprocessing_test(test_X)

    # X, y = preprocessing_train(pd.read_csv("../data/train.feats.csv"), pd.read_csv("../data/train.labels.1.csv"), multi_label=False)
    # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=42)
    preds = []
    rfc = LinearRegression()
    rfc.fit(train_X, train_y)
    # preds.append(lr.predict(test_X))
    # print(sklearn.metrics.mean_squared_error(pred, test_y))

    est = RFE(rfc, n_features_to_select=36)
    est.fit(train_X, train_y)
    preds.append(est.predict(test_X))
    gold = pd.DataFrame(test_y)
    # print(sklearn.metrics.mean_squared_error(pred, test_y))

    rfr = RandomForestRegressor(n_estimators=7)
    rfr.fit(train_X, train_y)
    preds.append(rfr.predict(test_X))

    brg = BaggingRegressor(n_estimators=7)
    brg.fit(train_X, train_y)
    preds.append(brg.predict(test_X))

    for p in preds:
        p[p<0] = 0

    pred = np.mean(preds, axis=0)
    y_pred = pd.DataFrame()
    y_gold = pd.DataFrame()
    y_pred["prediction"] = pred
    y_gold["prediction"] = test_y[test_y.columns[0]]
    y_pred["prediction"].to_csv("./tumor_size_pred.csv", header=False, index=False)
    y_gold["prediction"].to_csv("./tumor_size_gold.csv", header=False, index=False)
    print(sklearn.metrics.mean_squared_error(y_gold, y_pred))
    pred = est.predict(test_X)
    pd.DataFrame(pred).to_csv("./y_pred_tumur_size.csv", header=["Tumor_size"], index=False)
    gold = pd.DataFrame(test_y["??????????-Tumor size"])
    gold.to_csv("./y_gold_tumor_size.csv", header=["Tumor_size"], index=False)
    print(sklearn.metrics.mean_squared_error(pred, test_y))

    rf_tumor_size = RandomForestRegressor(n_estimators=10)
    rf_tumor_size.fit(train_X, train_y)

    # print(f"Num of features: {X.shape[1]}")
    # features_arr = list(range(1, 45))
    # errs = [10]
    # for i in features_arr:
    #     est = RFE(lr, n_features_to_select=i)
    #     est.fit(train_X, train_y)
    #     pred = est.predict(test_X)
    #     errs.append(sklearn.metrics.mean_squared_error(pred, test_y))
    # print(f"Best error: {errs[np.argmin(errs)]} is for {np.argmin(errs)} num of features")


def _doc_(args):
    pass


if __name__ == '__main__':
    args = docopt(__doc__)

    # Parse command line arguments

    train_features = Path(args["--train_feat"])
    test_features = Path(args["--test_feat"])
    labels_0 = Path(args["--labels0"])
    labels_1 = Path(args["--labels1"])
    out0 = Path(args["--labels0"])
    out1 = Path(args["--labels1"])

    # read data
    train_X_tumor = pd.read_csv(train_features)
    train_X_class = pd.read_csv(train_features)
    train_y_class = pd.read_csv(labels_0)
    train_y_tumor = pd.read_csv(labels_1)
    test_X = pd.read_csv(test_features)


    train_X_class, train_y_class = preprocessing_train(train_X_class, train_y_class)
    train_X_tumor, train_y_tumor = preprocessing_train(train_X_tumor, train_y_tumor, False)
    test_X = preprocessing_test(test_X)

    rfc = RandomForestClassifier()
    bgc = BaggingClassifier()
    y_pred = pd.DataFrame()
    
    for col in test_X.columns:
        if col not in train_X_class.columns:
            test_X.drop([col], inplace=True, axis=1)
    
    for column in train_y_class.columns:
        rfc.fit(train_X_class, train_y_class[column])
        bgc.fit(train_X_class, train_y_class[column])
        preds = [rfc.predict(test_X), bgc.predict(test_X)]
        y_pred[column] = np.max(preds, axis=0)
    

    y_pred["prediction"] = y_pred.apply(lambda x: str([c for c in x.index if x[c] > 0]), axis=1)
    y_pred["prediction"].to_csv("../task2/part1/predictions.csv", header=False, index=False)

    rfr = RandomForestRegressor(n_estimators=100, max_depth=8)
    rfr.fit(train_X_tumor, train_y_tumor)

    pred = rfr.predict(test_X)
    y_pred = pd.DataFrame()
    y_pred["prediction"] = pred
    y_pred["prediction"].to_csv("../task2/part2/predictions.csv", header=False, index=False)
