import pandas as pd
import re

def string2boolean(string):
    POSITIVE_STRINGS = {"pos", "+", "extensive", "micropapillary variant", "yes", "(+)"}
    NEGATIVE_STRINGS = {"none", "-", "no", "(-)", "neg", "not"}
    if pd.isna(string) or string.lower() in NEGATIVE_STRINGS:
        return 0
    elif string.lower() in POSITIVE_STRINGS:
        return 1
    return string 


def histopatological_degree_to_int(degree: str):
    values = ["null", "gx", "g1", "g2", "g3", "g4"]
    for idx, val in enumerate(values):
        if val in degree.lower():
            return idx
    return 0


def preprocessing(df: pd.DataFrame):
    df = df.rename(columns={col: re.sub(r'[^\x00-\x7F]+','', col).strip().replace(' ','_').replace('-','') for col in df.columns})
    # Convert Ivi_Lymphovascular_invasion to boolean
    df["Ivi_Lymphovascular_invasion"] = df["Ivi_Lymphovascular_invasion"].apply(string2boolean)
    # Convert Histopatological degree to int (greater should be more correlated to cancer)
    df["Histopatological degree"] = df["Histopatological degree"].apply(histopatological_degree_to_int)
    # Num of surgeries in int (Na = 0)
    df['Surgery sum'] = df['Surgery sum'].fillna(0)


if __name__ == "__main__":
    preprocessing(pd.read_csv("data/train.feats.csv"))