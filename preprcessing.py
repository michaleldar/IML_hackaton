import pandas as pd
import re


def string2boolean(string, default=0):
    POSITIVE_STRINGS = {"pos", "+", "extensive", "micropapillary variant", "yes", "(+)", "חיובי", "jhuch"}
    NEGATIVE_STRINGS = {"none", "-", "no", "(-)", "neg", "not", "שלילי", "akhkh"}
    if pd.isna(string) or string.lower() in NEGATIVE_STRINGS or any(s in string.lower() for s in NEGATIVE_STRINGS):
        return 0
    elif string.lower() in POSITIVE_STRINGS or any(s in string.lower() for s in POSITIVE_STRINGS):
        return 1
    return default 


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


def tumor_mark(string):
    values = ["tx", "tis", "t0", "t1", "t1mic", "t1a", "t1b", "t1c", "t1d", "t2", "t2a", "t2b", "t2c",
              "t2d", "t3", "t3a", "t3b", "t3c", "t3d", "t4", "t4a", "t4b", "t4c", "t4d"]
    if string is None: return -1
    string = string.lower()
    for idx, val in enumerate(values):
        if string == val:
            return idx
    last_idx = -1
    for idx, val in enumerate(values):
        if val in string:
            last_idx = idx
    return last_idx


def lymph_nodes_mark(string):
    values = ["n", "n1", "n1a", "n1b", "n1c", "n1d", "n2", "n2a", "n2b", "n2c", "n2d", "n3", "n3a",
              "n3b", "n3c", "n3d", "n3d", "n4", "n4a", "n4b", "n4c", "n4d"]
    if not string:
        return -1
    string = string.lower()
    for idx, val in enumerate(values):
        if string == val:
            return idx
    last_idx = -1
    for idx, val in enumerate(values):
        if val in string:
            last_idx = idx
    return last_idx


def metastases_mark(string):
    values = ["m0", "mo", "m1", "m1a", "m1b"]
    if not string:
        return -1
    string = string.lower()
    for idx, val in enumerate(values):
        if string == val:
            return idx
    last_idx = -1
    for idx, val in enumerate(values):
        if val in string:
            last_idx = idx
    return last_idx



def histological_diagnosis_invasive(string):
    words = string.lower().split(" ")
    words = [w.replace(',', '') for w in words]
    if "carcinoma" in string.lower():
        if "in situ" in words or "intraductal" in words:
            return 0
        if "nos" in words:
            return 0.5
        return 1
    return 0


def histological_diagnosis_noninvasive(string):
    words = string.lower().split(" ")
    words = [w.replace(',', '') for w in words]
    if "carcinoma" in string.lower():
        if "in situ" in words or "intraductal" in words:
            return 1
        if "nos" in words:
            return 0.5
        return 0
    return 0


def preprocessing(df: pd.DataFrame):

    # Standardize column names
    df = df.rename(columns={col: re.sub(r'[^\x00-\x7F]+','', col).strip().replace(' ','_').replace('-','') for col in df.columns})
    # Remove duplicate entries - leave one row per patient and date
    df = pd.get_dummies(df, columns=["Form_Name"])
    df = df.groupby(by=['idhushed_internalpatientid']).first()

    columns_to_remove = ["Histological_diagnosis", "Form_Name"]

    # Convert Ivi_Lymphovascular_invasion to boolean
    df["Ivi_Lymphovascular_invasion"] = df["Ivi_Lymphovascular_invasion"].apply(string2boolean)
    # Convert Histopatological degree to int (greater should be more correlated to cancer)
    df["Histopatological_degree"] = df["Histopatological_degree"].apply(histopatological_degree_to_int)
    # Num of surgeries in int (Na = 0)
    df['Surgery_sum'] = df['Surgery_sum'].fillna(0)

    df["Her2"] = df["Her2"].apply(string2boolean)

    df = df[(df.Age > 0)]
    df = df[(df.Age < 120)]

    cancer_basic_stage_map = {'Null': 0, 'c - Clinical': 1, 'p - Pathological': 2, 'r - Reccurent': 3}
    df["Stage"] = df["Stage"].apply(clean_stage)
    df["Basic_stage"] = df["Basic_stage"].map(cancer_basic_stage_map)

    df["KI67_protein"] = df["KI67_protein"].apply(handle_ki67)
    df['KI67_protein'].fillna((df['KI67_protein'].mean()), inplace=True)

    df["Histological_diagnosis_invasive"] = df["Histological_diagnosis"].apply(histological_diagnosis_invasive)
    df["Histological_diagnosis_noninvasive"] = df["Histological_diagnosis"].apply(histological_diagnosis_noninvasive)

    df["T_Tumor_mark_(TNM)"] = df["T_Tumor_mark_(TNM)"].apply(tumor_mark)
    df["N_lymph_nodes_mark_(TNM)"] = df["N_lymph_nodes_mark_(TNM)"].apply(lymph_nodes_mark)
    df["M_metastases_mark_(TNM)"] = df["M_metastases_mark_(TNM)"].apply(metastases_mark)


if __name__ == "__main__":
    preprocessing(pd.read_csv("data/train.feats.csv"))