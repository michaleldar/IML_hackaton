import pandas as pd
import re
import ast
from sklearn.model_selection import train_test_split

LOCATIONS = {'ADR - Adrenals',
 'BON - Bones',
 'BRA - Brain',
 'HEP - Hepatic',
 'LYM - Lymph nodes',
 'MAR - Bone Marrow',
 'OTH - Other',
 'PER - Peritoneum',
 'PLE - Pleura',
 'PUL - Pulmonary',
 'SKI - Skin'}

one_hot_cols = ['Hospital', 'Margin_Type']
cols_to_drop = ['Margin_Type_without', 'User_Name', 'Side', 'Tumor_depth', 'Nodes_exam',
                'surgery_before_or_afterActivity_date', 'surgery_before_or_afterActual_activity']

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


def date_diff(str1, str2):
    if str1 is None or str2 is None:
        return -1
    from dateutil.parser import parse
    try:
        dt1 = parse(str1, fuzzy=False)
        dt2 = parse(str2, fuzzy=False)
        return abs((dt1 - dt2).days)

    except ValueError:
        return -1


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


def BIOPSY_surgery(string):
    if not string:
        return 0
    if "biopsy" in string.lower():
        return 1
    return 0


def LUMPECTOMY_surgery(string):
    if not string:
        return 0
    if "lumpectomy" in string.lower() or "excision" in string.lower() or "exc." in string.lower():
        return 1
    return 0


def MASTECTOMY_surgery(string):
    if not string:
        return 0
    if "mastectomy" in string.lower():
        return 1
    return 0


def QUADRANTECTOMY_surgery(string):
    if not string:
        return 0
    if "quadrantectomy" in string.lower():
        return 1
    return 0


def OOPHORECTOMY_surgery(string):
    if not string:
        return 0
    if "oophorectomy" in string.lower():
        return 1
    return 0


def divide_nodes(positive, exam):
    if exam == 0: return 0
    return positive / exam


def clean_er_pr(string):
    if pd.isna(string): return 0
    string = string.lower()
    if "po" in string: return 1
    if "strong" in string: return 1
    if "high" in string: return 1
    if "חיובי" in string: return 1
    if "neg" in string: return 0
    if "+" in string: return 1
    if string == "-": return 0
    if string == "(-)": return 0
    if "<1" in string: return 0
    floats = [float(s) for s in re.findall(r"[+-]? (?:\d+(?:\.\d)?|\.\d+)(?:[eE][+-]?\d+)?", string)]
    if len(floats) == 0: return 0
    if floats[0] > 0: return 1
    return 0

def string2set(string):
    lst = ast.literal_eval(string)
    return set(lst)

def preprocessing(df: pd.DataFrame, labels: pd.DataFrame):

    # Standardize column names
    df = df.rename(columns={col: re.sub(r'[^\x00-\x7F]+','', col).strip().replace(' ','_').replace('-','') for col in df.columns})
    labels = labels.rename(columns={col: re.sub(r'[^\x00-\x7F]+','', col).strip().replace(' ','_').replace('-','') for col in labels.columns})
    # Remove duplicate entries - leave one row per patient and date

    form_name_map = {'אנמנזה סיעודית': 'nursing_anamnesis',
                     'אומדן סימפטומים ודיווח סיעודי': 'symptoms_eval_nursing_report',
                     'אנמנזה רפואית': 'medical_anamnesis', 'אנמנזה סיעודית קצרה': 'nursing_anamnesis_short',
                     'ביקור במרפאה': 'clinic_visit', 'אנמנזה רפואית המטו-אונקולוגית': 'onco_anamnesis',
                     'ביקור במרפאה קרינה': 'radiation_clinic_visit',
                     'ביקור במרפאה המטו-אונקולוגית': 'onco_clinic_visit'}
    df["Form_Name"] = df["Form_Name"].map(form_name_map)
    df = pd.get_dummies(df, columns=["Form_Name"])

    df["Location_of_distal_metastases"] = labels["Location_of_distal_metastases"]
    df = df.groupby(by=['idhushed_internalpatientid']).first()
    labels = df["Location_of_distal_metastases"]

    # Turn labels into matrix
    labels = labels.apply(string2set)
    labels_mat = pd.DataFrame()
    for loc in LOCATIONS:
        indicator = lambda x: 1 if loc in x else 0
        labels_mat[loc] = labels.apply(indicator)

    df.drop(["Location_of_distal_metastases"], inplace=True, axis=1)


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

    df['Surgery_date1_diff'] = df.apply(lambda x: date_diff(x.Surgery_date1, x.Diagnosis_date), axis=1)
    df['Surgery_date2_diff'] = df.apply(lambda x: date_diff(x.Surgery_date2, x.Diagnosis_date), axis=1)
    df['Surgery_date3_diff'] = df.apply(lambda x: date_diff(x.Surgery_date3, x.Diagnosis_date), axis=1)

    # first surgery name
    df["BIOPSY_surgery_1"] = df["Surgery_name1"].apply(BIOPSY_surgery)
    df["LUMPECTOMY_surgery_1"] = df["Surgery_name1"].apply(LUMPECTOMY_surgery)
    df["MASTECTOMY_surgery_1"] = df["Surgery_name1"].apply(MASTECTOMY_surgery)
    df["QUADRANTECTOMY_surgery_1"] = df["Surgery_name1"].apply(QUADRANTECTOMY_surgery)
    df["OOPHORECTOMY_surgery_1"] = df["Surgery_name1"].apply(OOPHORECTOMY_surgery)

    # second surgery name
    df["BIOPSY_surgery_2"] = df["Surgery_name2"].apply(BIOPSY_surgery)
    df["LUMPECTOMY_surgery_2"] = df["Surgery_name2"].apply(LUMPECTOMY_surgery)
    df["MASTECTOMY_surgery_2"] = df["Surgery_name2"].apply(MASTECTOMY_surgery)
    df["QUADRANTECTOMY_surgery_2"] = df["Surgery_name2"].apply(QUADRANTECTOMY_surgery)
    df["OOPHORECTOMY_surgery_2"] = df["Surgery_name2"].apply(OOPHORECTOMY_surgery)

    # third surgery name
    df["BIOPSY_surgery_3"] = df["Surgery_name3"].apply(BIOPSY_surgery)
    df["LUMPECTOMY_surgery_3"] = df["Surgery_name3"].apply(LUMPECTOMY_surgery)
    df["MASTECTOMY_surgery_3"] = df["Surgery_name3"].apply(MASTECTOMY_surgery)
    df["QUADRANTECTOMY_surgery_3"] = df["Surgery_name3"].apply(QUADRANTECTOMY_surgery)
    df["OOPHORECTOMY_surgery_3"] = df["Surgery_name3"].apply(OOPHORECTOMY_surgery)



    lymphatic_penetration_map = {'L0 - No Evidence of invasion': 0, 'Null': 0, 'LI - Evidence of invasion': 1,
                                 'L1 - Evidence of invasion of superficial Lym.': 2,
                                 'L2 - Evidence of invasion of depp Lym.': 3}
    margin_type_map = {'ללא': 'without', 'נקיים': 'clean', 'נגועים': 'contaminated'}
    df["Lymphatic_penetration"] = df["Lymphatic_penetration"].map(lymphatic_penetration_map)
    df["Margin_Type"] = df["Margin_Type"].map(margin_type_map)

    df = pd.get_dummies(df, columns=one_hot_cols)

    # Tumor side
    df["Side_left"] = df["Side"].isin({"שמאל", "דו צדדי"}).astype(int)
    df["Side_right"] = df["Side"].isin({"ימין", "דו צדדי"}).astype(int)

    # Tumor width
    df["Tumor_width"] = df["Tumor_width"].fillna(0)

    # Positive nodes ratio
    df["Nodes_exam"] = df["Nodes_exam"].fillna(0)
    df["Positive_nodes"] = df["Positive_nodes"].fillna(0)
    df['Positive_node_rate'] = df.apply(lambda x: divide_nodes(x.Positive_nodes, x.Nodes_exam), axis=1)

    # ER and PR
    df['er'] = df['er'].apply(clean_er_pr)
    df['pr'] = df['pr'].apply(clean_er_pr)

    return df, labels_mat


if __name__ == "__main__":
    X, y = preprocessing(pd.read_csv("data/train.feats.csv"), pd.read_csv("data/train.labels.0.csv"))