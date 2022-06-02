import sklearn
import pandas as pd
import numpy as np
import plotly.express as px
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import re

class preprocessor():
    def __init__(self, data_file_name: str, lables_file_name: str):
        self.df = pd.read_csv(r"https://github.com/michahashkes/IMLHackathon/blob/main/data/train_data.csv?raw=true")
        self.labels = pd.read_csv(r"https://github.com/michahashkes/IMLHackathon/blob/main/data/train_labels.csv?raw=true")

        self.labels['locationDistalMetastases'] = self.labels['locationDistalMetastases'].apply(ast.literal_eval)
        mlb = MultiLabelBinarizer().fit(self.labels['locationDistalMetastases'])
        self.lables_binary = mlb.transform(self.labels['locationDistalMetastases'])
        self.lables_binary = pd.DataFrame(self.lables_binary, columns=mlb.classes_)
        self.lables_binary["sum"] = self.lables_binary.sum(axis=1)

    def age(self):
        """
        no changes in age column
        """
        return

    def basicStage(self):
        """
        make ordinal categories
        """
        basicStageDict = {"Null": 0, "c - Clinical": 1, "p - Pathological": 2, "r - Reccurent": 3}
        self.df["basicStage"] = self.df["basicStage"].apply(lambda x: basicStageDict[x] if x in basicStageDict else 0)

    def histologicalDiagnosis(self):
        """
        make dummies
        """
        self.df = pd.get_dummies(self.df, columns=['histologicalDiagnosis'])

    def lymphovascularInvasion(self):
        """
        replace positive values with 1, neg values with 0 -> BASED ON TRAIN DATA!!!
        """
        positive_values = ["yes", "pos", "(+)", "+", "YES", "Y", "y", "extensive", "MICROPAPILLARY VARIANT", "positive"]
        self.df["lymphovascularInvasion"] = self.df["lymphovascularInvasion"].apply(lambda x: 1 if x in positive_values else 0)

    def lymphNodesMarkTNM(self):
        """
        make ordinal categories
        N0 - it is known that there is no cancer in lymph nodes
        NX - there is no information. so, Null values are classified as NX.
        """
        TNM_N_stages_dict = {"N0": 0, "NX": 1, "N1": 2, "N1a": 2, "N1mic": 2, "N1b": 3, "N1c": 4, "N1d": 5, "N2": 6,
                             "N2a": 7, "N2b": 8, "N2c": 9, "N2d": 10, "N3": 11, "N3a": 11, "N3b": 12, "N3c": 13,
                             "N3d": 14, "N4": 15, "N4a": 15, "N4b": 16, "N4c": 17, "N4d": 18}
        self.df["lymphNodesMarkTNM"] = self.df["lymphNodesMarkTNM"].apply(lambda x: TNM_N_stages_dict[x] if x in TNM_N_stages_dict else 1)

    def stage(self):
        """
        make ordinal categories.
        None - get median values
        """
        Stages_dict = {"Stage0": 1, "Stage0is": 1, "Stage1": 2, "Stage1a": 2, "Stage1b": 3, "Stage1c": 4, "Stage1d": 5,
                       "Stage2": 6, "Stage2a": 6, "Stage2b": 7, "Stage2c": 8, "Stage2d": 9, "Stage3": 10, "Stage3a": 10,
                       "Stage3b": 11, "Stage3c": 12, "Stage3d": 13, "Stage4": 14, "Stage4a": 14, "Stage4b": 15,
                       "Stage4c": 16, "Stage4d": 17}
        self.df["stage"] = self.df["stage"].apply(lambda x: Stages_dict[x] if x in Stages_dict else None)
        self.df["stage"].fillna(value=self.df.stage.median(), inplace=True)

    def surgerySum(self):
        """
        fillna with 0 - no surgeries
        """
        self.df["surgerySum"].fillna(value=0, inplace=True)

    def surgeryNames(self):
        """
        get all surgeries' names, make them dummies, and delete all 3 cols of surgery names (1,2 and 3)
        """
        temp = self.df[["surgeryName1", "surgeryName2", "surgeryName3"]].stack().str.get_dummies().sum(level=0)
        self.df = self.df.join(temp)
        self.df[temp.columns] = self.df[temp.columns].fillna(0)

        self.df = self.df.drop(["surgeryName1", "surgeryName2", "surgeryName3"], 1)

    def get_percentage(input):
        ls = re.findall("[0-9]+%", input)
        ls = [int(sub.replace('%', '')) for sub in ls]
        return np.mean(ls)

    def KI67_protein(self):
        """

        """
        self.df["KI67_protein"] = self.df["KI67_protein"].fillna(-1).astype(str).apply(self.get_percentage)
        self.df["KI67_protein"] = np.where(self.df["KI67_protein"] > 100, np.nan, self.df["KI67_protein"])

        self.df["KI67_protein"] = self.df["KI67_protein"].fillna(np.mean(self.df["KI67_protein"]))

    def lymphaticPen(self):
        """
        unnecessary data, dropping
        """
        self.df = self.df.drop("lymphaticPen", 1)

    def mMetastasesMarkTNM(self):
        """
        similar to TNM N
        """
        TNM_M_stages_dict = {"M0": 0, "M1": 2, "MX": 1, "Not yet Established":1, "M1a":2, "M1b":3}
        self.df["mMetastasesMarkTNM"] = self.df["mMetastasesMarkTNM"].apply(
            lambda x: TNM_M_stages_dict[x] if x in TNM_M_stages_dict else 1)

    def marginType(self):

        self.df = pd.get_dummies(self.df, columns=["marginType"])
        self.df = self.df.drop("marginType_none", 1)

    def
        Stages_dict = {"T0": 0, "Tis": 1, "T1mic": 2, "T1a": 2, "T1b": 3, "T1": 3, "T1c": 3, "T2a": 4, "T2": 4,
                       "T2b": 4, "T3b": 5, "T3": 5, "T3c": 6, "T3d": 6, "T4a": 7, "T4": 8, "T4c": 9, "T4d": 9}
        df["tumorMarkTNM"] = df["אבחנה-T -Tumor mark (TNM)"].apply(
            lambda x: Stages_dict[x] if x in Stages_dict else None)


