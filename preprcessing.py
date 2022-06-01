import pandas as pd
import re

def string2boolean(string):
    POSITIVE_STRINGS = {"pos", "+", "extensive", "micropapillary variant", "yes", "(+)"}
    NEGATIVE_STRINGS = {"none", "-", "no", "(-)", "neg", "not"}
    if pd.isna(string) or string in NEGATIVE_STRINGS:
        return 0
    elif string in POSITIVE_STRINGS:
        return 1
    return string 


def preprocessing(df: pd.DataFrame):
    df = df.rename(columns={col: re.sub(r'[^\x00-\x7F]+','', col).strip().replace(' ','_').replace('-','') for col in df.columns})
    # Convert Ivi_Lymphovascular_invasion to boolean
    df["Ivi_Lymphovascular_invasion"] = df["Ivi_Lymphovascular_invasion"].apply(string2boolean)

if __name__ == "__main__":
    preprocessing(pd.read_csv("data/train.feats.csv"))