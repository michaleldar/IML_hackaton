import pandas as pd


def histopatological_degree_to_int(degree: str):
    values = ["null", "gx", "g1", "g2", "g3", "g4"]
    for idx, val in enumerate(values):
        if val in degree.lower():
            return idx
    return 0


def preprocessing(df: pd.DataFrame):
    pass


if __name__ == "__main__":
    preprocessing(pd.read_csv("data/train.feats.csv"))