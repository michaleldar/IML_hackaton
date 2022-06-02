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

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    from dateutil.parser import parse
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def handle_ki67(string):
    if type(string) != str: return string
    if is_date(string): return pd.NA

    # Find largest number in string
    nums = [int(x) for x in re.findall(r'\d+', string)]
    if len(nums) == 0: return pd.NA
    return max(nums)


def histopatological_degree_to_int(degree: str):
    values = ["null", "gx", "g1", "g2", "g3", "g4"]
    for idx, val in enumerate(values):
        if val in degree.lower():
            return idx
    return 0

def clean_stage(string):
    if string == 'LA': return -1
    if string == 'Not yet Established': return 0
    if pd.isna(string): return 0
    return int(re.sub("[a-zA-Z]+", "", string))


def preprocessing(df: pd.DataFrame):

    # Remove duplicate entries - leave one row per patient and date
    df = pd.get_dummies(df, prefix=["Form_Name"])
    df = df.groupby(by=['Diagnosis_date', 'idhushed_internalpatientid']).first()


    df = df.rename(columns={col: re.sub(r'[^\x00-\x7F]+','', col).strip().replace(' ','_').replace('-','') for col in df.columns})
    # Convert Ivi_Lymphovascular_invasion to boolean
    df["Ivi_Lymphovascular_invasion"] = df["Ivi_Lymphovascular_invasion"].apply(string2boolean)
    # Convert Histopatological degree to int (greater should be more correlated to cancer)
    df["Histopatological degree"] = df["Histopatological degree"].apply(histopatological_degree_to_int)
    # Num of surgeries in int (Na = 0)
    df['Surgery sum'] = df['Surgery sum'].fillna(0)

    df = df[(df.Age > 0)]
    df = df[(df.Age < 120)]

    cancer_basic_stage_map = {'Null': 0, 'c - Clinical': 1, 'p - Pathological': 2, 'r - Reccurent': 3}
    df["Stage"] = df["Stage"].apply(clean_stage)
    df["Basic_stage"] = df["Basic_stage"].map(cancer_basic_stage_map)

    df["KI67_protein"] = df["KI67_protein"].apply(handle_ki67)
    df['KI67_protein'].fillna((df['KI67_protein'].mean()), inplace=True)

if __name__ == "__main__":
    preprocessing(pd.read_csv("data/train.feats.csv"))