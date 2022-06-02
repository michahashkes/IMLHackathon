import sklearn
import pandas as pd
import numpy as np
import plotly.express as px
import ast
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import re
from datetime import datetime

new_columns = {' Form Name': 'formName',
               ' Hospital': 'hospital',
               'User Name': 'userName',
               'אבחנה-Age': 'age',
               'אבחנה-Basic stage': 'basicStage',
               'אבחנה-Diagnosis date': 'diagnosisDate',
               'אבחנה-Her2': 'her2',
               'אבחנה-Histological diagnosis': 'histologicalDiagnosis',
               'אבחנה-Histopatological degree': 'histopatologicalDegree',
               'אבחנה-Ivi -Lymphovascular invasion': 'lymphovascularInvasion',
               'אבחנה-KI67 protein': 'KI67_protein',
               'אבחנה-Lymphatic penetration': 'lymphaticPenetration',
               'אבחנה-M -metastases mark (TNM)': 'mMetastasesMarkTNM',
               'אבחנה-Margin Type': 'marginType',
               'אבחנה-N -lymph nodes mark (TNM)': 'lymphNodesMarkTNM',
               'אבחנה-Nodes exam': 'nodesExam',
               'אבחנה-Positive nodes': 'positiveLymph',
               'אבחנה-Side': 'side',
               'אבחנה-Stage': 'stage',
               'אבחנה-Surgery date1': 'surgeryDate1',
               'אבחנה-Surgery date2': 'surgeryDate2',
               'אבחנה-Surgery date3': 'surgeryDate3',
               'אבחנה-Surgery name1': 'surgeryName1',
               'אבחנה-Surgery name2': 'surgeryName2',
               'אבחנה-Surgery name3': 'surgeryName3',
               'אבחנה-Surgery sum': 'surgerySum',
               'אבחנה-T -Tumor mark (TNM)': 'tumorMarkTNM',
               'אבחנה-Tumor depth': 'tumorDepth',
               'אבחנה-Tumor width': 'tumorWidth',
               'אבחנה-er': 'er',
               'אבחנה-pr': 'pr',
               'surgery before or after-Activity date': 'activityDate',
               'surgery before or after-Actual activity': 'actualActivity',
               'id-hushed_internalpatientid': 'id'}


class Preprocessor:
    def __init__(self, data_file_name: str, labels0_file_name: str, labels1_file_name: str, train=True,
                 mlb: MultiLabelBinarizer = None, form_names_enc: OneHotEncoder = None,
                 histological_diagnosis_enc: OneHotEncoder = None, margin_type_enc: OneHotEncoder = None,
                 surgery_name_enc: MultiLabelBinarizer = None):
        self.df = pd.read_csv(data_file_name)

        self.df.rename(columns=new_columns, inplace=True)

        self.mlb, self.form_names_enc, self.histological_diagnosis_enc, self.margin_type_enc, self.surgery_name_enc, = \
            mlb, form_names_enc, histological_diagnosis_enc, margin_type_enc, surgery_name_enc
        self.train = train

        if self.train:
            self.labels0 = pd.read_csv(labels0_file_name)
            self.labels0.rename(columns={'אבחנה-Location of distal metastases': 'locationDistalMetastases'}, inplace=True)
            self.labels1 = pd.read_csv(labels1_file_name)
            X_y = self.df.join(self.labels0)
            X_y = X_y.join(self.labels1)
            forms_df = X_y[['id', 'formName']]
            self.form_names_enc = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(forms_df[['formName']])
            encoded_columns = pd.DataFrame(self.form_names_enc.transform(forms_df[['formName']]),
                                           columns=self.form_names_enc.categories_)
            forms_df = forms_df.join(encoded_columns).drop(columns='formName')
            forms_df = forms_df.groupby('id').max().reset_index()
            X_y = X_y.merge(forms_df, on='id')
            X_y = X_y.drop(columns=['userName', 'formName']).drop_duplicates().reset_index(drop=True)
            self.df = X_y.drop(columns=['locationDistalMetastases', 'אבחנה-Tumor size'])
            self.labels0 = X_y[['locationDistalMetastases']]
            self.labels1 = X_y['אבחנה-Tumor size']

        else:
            forms_df = self.df[['id', 'formName']]
            encoded_columns = pd.DataFrame(self.form_names_enc.transform(forms_df[['formName']]),
                                           columns=self.form_names_enc.categories_)
            self.df = self.df.join(encoded_columns).drop(columns=['userName', 'formName'])
        # self.labels0.rename(columns={'אבחנה-Location of distal metastases': 'locationDistalMetastases'}, inplace=True)

    def fix_data(self, train=True):
        X_y = self.df.join(self.labels0)
        X_y = X_y.join(self.labels1)

        forms_df = X_y[['id', 'formName']]
        if self.form_names_enc is None:
            self.form_names_enc = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(forms_df[['formName']])
        encoded_columns = pd.DataFrame(self.form_names_enc.transform(forms_df[['formName']]),
                                       columns=self.form_names_enc.categories_)
        # forms_df = forms_df.join(encoded_columns)
        # forms_df = pd.get_dummies(forms_df, columns=['formName'])

        if train:
            forms_df = forms_df.groupby('id').max().reset_index()
            X_y = X_y.merge(forms_df, on='id')
            X_y = X_y.drop(columns=['userName', 'formName']).drop_duplicates().reset_index(drop=True)
        else:
            X_y = X_y.join(encoded_columns)

        X = X_y.drop(columns=['אבחנה-Location of distal metastases', 'אבחנה-Tumor size'])
        y0 = X_y['אבחנה-Location of distal metastases']
        y1 = X_y['אבחנה-Tumor size']
        return X, y0, y1

    def binarize_labels(self):
        self.labels0['locationDistalMetastases'] = self.labels0['locationDistalMetastases'].apply(ast.literal_eval)
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit(self.labels0['locationDistalMetastases'])
        self.lables_binary = self.mlb.transform(self.labels0['locationDistalMetastases'])
        self.lables_binary = pd.DataFrame(self.lables_binary, columns=self.mlb.classes_)
        # self.lables_binary["sum"] = self.lables_binary.sum(axis=1)

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
        if self.histological_diagnosis_enc is None:
            self.histological_diagnosis_enc = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(self.df[['histologicalDiagnosis']])
        encoded_columns = pd.DataFrame(self.histological_diagnosis_enc.transform(self.df[['histologicalDiagnosis']]),
                                       columns=self.histological_diagnosis_enc.categories_)
        self.df = self.df.join(encoded_columns)
        self.df = self.df.drop(columns="histologicalDiagnosis")

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
        # surgery1 = self.df["surgeryName1"].fillna('')
        # surgery2 = self.df["surgeryName2"].fillna('')
        # surgery3 = self.df["surgeryName3"].fillna('')
        # surgery = surgery1
        combined = pd.Series(self.df[["surgeryName1", "surgeryName2", "surgeryName3"]].values.tolist())
        combined = combined.apply(lambda l: [x for x in l if not pd.isnull(x)])
        if self.surgery_name_enc is None:
            self.surgery_name_enc = MultiLabelBinarizer()
            self.surgery_name_enc.fit(combined)
        combined = self.surgery_name_enc.transform(combined)
        combined = pd.DataFrame(combined, columns=self.surgery_name_enc.classes_)

        # temp = self.df[["surgeryName1", "surgeryName2", "surgeryName3"]].stack()#.str#.get_dummies().sum(level=0)
        self.df = self.df.join(combined)
        # self.df[temp.columns] = self.df[temp.columns].fillna(0)

        self.df = self.df.drop(columns=["surgeryName1", "surgeryName2", "surgeryName3"])

    def KI67_protein(self):
        """

        """
        self.df["KI67_protein"] = self.df["KI67_protein"].fillna(-1).astype(str).apply(self.get_percentage)
        self.df["KI67_protein"] = np.where(self.df["KI67_protein"] > 100, np.nan, self.df["KI67_protein"])

        self.df["KI67_protein"] = self.df["KI67_protein"].fillna(np.mean(self.df["KI67_protein"]))

    def lymphaticPenetration(self):
        """
        unnecessary data, dropping
        """
        self.df = self.df.drop(columns="lymphaticPenetration")

    def mMetastasesMarkTNM(self):
        """
        similar to TNM N
        """
        TNM_M_stages_dict = {"M0": 0, "M1": 2, "MX": 1, "Not yet Established":1, "M1a":2, "M1b":3}
        self.df["mMetastasesMarkTNM"] = self.df["mMetastasesMarkTNM"].apply(
            lambda x: TNM_M_stages_dict[x] if x in TNM_M_stages_dict else 1)

    def marginType(self):
        if self.margin_type_enc is None:
            self.margin_type_enc = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(self.df[['marginType']])
        encoded_columns = pd.DataFrame(self.margin_type_enc.transform(self.df[['marginType']]),
                                       columns=self.margin_type_enc.categories_)
        self.df = self.df.join(encoded_columns)
        self.df = self.df.drop(columns="marginType")

    def tumorMarkTNM(self):

        Stages_dict = {"T0": 0, "Tis": 1, "T1mic": 2, "T1a": 2, "T1b": 3, "T1": 3, "T1c": 3, "T2a": 4, "T2": 4,
                       "T2b": 4, "T3b": 5, "T3": 5, "T3c": 6, "T3d": 6, "T4a": 7, "T4": 8, "T4c": 9, "T4d": 9}
        self.df["tumorMarkTNM"] = self.df["tumorMarkTNM"].apply(
            lambda x: Stages_dict[x] if x in Stages_dict else None)

        self.df["tumorMarkTNM"].fillna(np.mean(self.df["tumorMarkTNM"]), inplace=True)

    def tumorDepth(self):
        self.df = self.df.drop(columns="tumorDepth")

    def tumorWidth(self):
        self.df = self.df.drop(columns="tumorWidth")

    def activityDate(self):
        self.df = self.df.drop(columns="activityDate")

    def actualActivity(self):
        self.df = self.df.drop(columns="actualActivity")

    def pr(self):

        self.df["pr_perc"] = self.df["pr"].fillna(-1).astype(str).apply(self.get_percentage)
        self.df["pr_perc"] = np.where(self.df["pr_perc"] < 10, 1, self.df["pr_perc"])
        self.df["pr_perc"] = np.where(self.df["pr_perc"] >= 10, 2, self.df["pr_perc"])

        self.df["pr_pos"] = self.df["pr"].fillna(99).astype(str).apply(self.get_sign)
        Stages_dict = {"-": 0, "n": 0, "N": 0, "+": 2, "p": 2, "P": 2}
        self.df["pr_pos"] = self.df["pr_pos"].apply(lambda x: Stages_dict[x] if x in Stages_dict else None)
        self.df["pr"] = np.where(self.df["pr_perc"].isnull(), self.df["pr_pos"], self.df["pr_perc"])
        self.df["pr"].fillna(1, inplace=True)
        self.df.drop(columns=["pr_pos", "pr_perc"], inplace=True)

    def histopatologicalDegree(self):
        Stages_dict = {"Null": 2, "GX - Grade cannot be assessed": 2, "G1 - Well Differentiated": 4, "G2 - Modereately well differentiated": 3,
                       "G3 - Poorly differentiated": 1, "G4 - Undifferentiated":0}
        self.df["histopatologicalDegree"] = self.df["histopatologicalDegree"].apply(lambda x: Stages_dict[x] if x in Stages_dict else 2)

    def nodesExam(self):
        self.df['nodesExam'].fillna(0, inplace=True)

    def positiveLymph(self):
        self.df["positiveLymph"].fillna(np.mean(self.df["positiveLymph"]), inplace=True)

    def surgeryDate(self):
        self.df["surgeryDate1"] = pd.to_datetime(self.df["surgeryDate1"], errors='coerce')
        self.df["surgeryDate2"] = pd.to_datetime(self.df["surgeryDate2"], errors='coerce')
        self.df["surgeryDate3"] = pd.to_datetime(self.df["surgeryDate3"], errors='coerce')

        # Create a column that indicates whether the patient has surgery
        self.df["hadSurgery"] = np.where(
            self.df["surgeryDate1"].isnull() & self.df["surgeryDate2"].isnull() & self.df["surgeryDate3"].isnull(), 0, 1)
        self.df["lastSurgeryDate"] = self.df[["surgeryDate1", "surgeryDate2", "surgeryDate2"]].max(axis=1)
        self.df.lastSurgeryDate.fillna(0, inplace=True)

        self.df.drop(columns=["surgeryDate1", "surgeryDate2", "surgeryDate3"], inplace=True)

    def er(self):
        self.df["er_perc"] = self.df["er"].fillna(-1).astype(str).apply(self.get_percentage)
        self.df["er_perc"] = np.where(self.df["er_perc"] < 10, 1, self.df["er_perc"])
        self.df["er_perc"] = np.where(self.df["er_perc"] >= 10, 2, self.df["er_perc"])

        self.df["er_pos"] = self.df["er"].fillna(99).astype(str).apply(self.get_sign)

        Stages_dict = {"-": 0, "n": 0, "N": 0, "+": 2, "p": 2, "P": 2}
        self.df["er_pos"] = self.df["er_pos"].apply(lambda x: Stages_dict[x] if x in Stages_dict else None)

        self.df["er"] = np.where(self.df["er_perc"].isnull(), self.df["er_pos"], self.df["er_perc"])
        self.df["er"].fillna(1, inplace=True)

        self.df.drop(columns=["er_perc", "er_pos"], inplace=True)

    def her2(self):
        her2value_dict = {'+3': 3, '+2': 2, '+1': 1, '+0': 0, '3+': 3, '2+': 2, '1+': 1, '0+': 0,
                          '0': 0, '1': 1, '2': 2, '3': 3,
                          'neg': 0, 'pos': 3, '-': 0, '+': 3, 'חיובי': 3, '?': 2, 'שלילי': 0,
                          'indeterm': 2, 'intermediate': 2, 'equivocal': 2, 'indet': 2, 'borderline': 2, 'nrg': 0,
                          'amplified': 3, 'akhah': 0, 'heg': 0, 'בינוני': 2, '_': 0, 'no': 0, 'nd': 0, 'akhkh': 0, 'nec': 0,
                          'pending': 2, 'nag': 0, 'po': 3}

        def get_her2(x):
            if pd.isnull(x):
                return 2
            for value in her2value_dict:
                if value in str(x).lower():
                    return her2value_dict[value]
            return 2

        self.df['her2'] = self.df['her2'].apply(get_her2)

    def side(self):
        self.df = pd.get_dummies(self.df, columns=['side'])
        # self.df['side'] = self.df['side'].replace({'שמאל': 0, 'ימין': 1})

    def get_sign(self, input):
        x = re.search("[pPnN+-]", input)
        if not x:
            return np.nan
        return input[x.start()]

    def get_percentage(self, input):
        ls = re.findall("[0-9]+%", input)
        ls = [int(sub.replace('%', '')) for sub in ls]
        return np.mean(ls)

    def preprocess(self):
        if self.train:
            self.binarize_labels()
        self.age()
        self.basicStage()
        self.histologicalDiagnosis()
        self.lymphovascularInvasion()
        self.lymphNodesMarkTNM()
        self.stage()
        self.surgerySum()
        self.surgeryNames()
        self.KI67_protein()
        self.lymphaticPenetration()
        self.mMetastasesMarkTNM()
        self.marginType()
        self.tumorMarkTNM()
        self.tumorWidth()
        self.tumorDepth()
        self.activityDate()
        self.actualActivity()
        self.pr()
        self.histopatologicalDegree()
        self.nodesExam()
        self.positiveLymph()
        self.surgeryDate()
        self.her2()
        self.side()
        self.er()

    def get_encoders(self):
        return self.mlb,  self.form_names_enc, self.histological_diagnosis_enc, self.margin_type_enc, self.surgery_name_enc

    def get_features(self):
        return self.df.columns

    def get_df(self):
        return self.df

    def get_labels0(self):
        return self.lables_binary

    def get_labels1(self):
        return self.labels1


if __name__ == '__main__':
    train_preprocessor = Preprocessor('../data/train.feats.csv', '../data/train.labels.0.csv', '../data/train.labels.1.csv')
    train_preprocessor.preprocess()
    train_df = train_preprocessor.get_df()
    train_labels = train_preprocessor.get_labels0()
    train_labels1 = train_preprocessor.get_labels1()

    encoders = train_preprocessor.get_encoders()
    test_preprocessor = Preprocessor('../data/test.feats.csv', '', '', False,
                                     encoders[0], encoders[1], encoders[2], encoders[3], encoders[4])
    test_preprocessor.preprocess()
    test_df = test_preprocessor.get_df()

    # train_preprocessor = Preprocessor('../data/train_data.csv', '../data/train_labels.csv', '../data/train_labels1.csv')
    # train_preprocessor.preprocess()
    # train_df = train_preprocessor.get_df()
    # train_labels = train_preprocessor.get_labels0()
    # train_labels1 = train_preprocessor.get_labels1()
    #
    # encoders = train_preprocessor.get_encoders()
    # test_preprocessor = Preprocessor('../data/dev_data.csv', '../data/dev_labels.csv', '../data/dev_labels1.csv', False,
    #                                  encoders[0], encoders[1], encoders[2], encoders[3], encoders[4])
    # test_preprocessor.preprocess()
    # test_df = test_preprocessor.get_df()
    # test_labels = test_preprocessor.get_labels0()
    # test_labels1 = test_preprocessor.get_labels1()

