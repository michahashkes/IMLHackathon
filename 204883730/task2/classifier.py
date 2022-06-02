import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from preprocessing import Preprocessor
from skmultilearn.adapt import MLkNN
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

def data_load():
    # replace paths if needed
    train_preprocessor = Preprocessor('../data/train.feats.csv', '../data/train.labels.0.csv', '../data/train.labels.1.csv')
    train_preprocessor.preprocess()
    train_df = train_preprocessor.get_df()
    train_labels1 = train_preprocessor.get_labels1()

    encoders = train_preprocessor.get_encoders()
    # replace path if needed
    test_preprocessor = Preprocessor('../data/test.feats.csv', '', '', False,
                                     encoders[0], encoders[1], encoders[2], encoders[3], encoders[4])
    test_preprocessor.preprocess()
    test_df = test_preprocessor.get_df()
    return train_df, train_labels1, test_df

def data_load(train_feats, train_label_zero, train_label_one, test_feats):
    # replace paths if needed
    train_preprocessor = Preprocessor(train_feats, train_label_zero, train_label_one)
    train_preprocessor.preprocess()
    train_df = train_preprocessor.get_df()
    train_labels0 = train_preprocessor.get_labels0()
    train_labels1 = train_preprocessor.get_labels1()

    encoders = train_preprocessor.get_encoders()
    # replace path if needed
    test_preprocessor = Preprocessor(test_feats, '', '', False,
                                     encoders[0], encoders[1], encoders[2], encoders[3], encoders[4])
    test_preprocessor.preprocess()
    test_df = test_preprocessor.get_df()
    return train_df, train_labels0, train_labels1, test_df

def train_predict(train_df, train_labels, test_df):
    regr = RandomForestRegressor(max_depth=4, random_state=0).fit(train_df, train_labels)
    return regr.predict(test_df)

def predict_tumor_size(train_df, train_labels, test_df):
    regr = RandomForestRegressor(max_depth=4, random_state=0).fit(train_df, train_labels)
    return regr.predict(test_df)

def predict_metastasis_location(train_df, train_labels, test_df):
    model = MLkNN(k=2)
    model.fit(train_df.values, train_labels.values)
    y_pred = model.predict(test_df).toarray()
    return pd.DataFrame(y_pred).apply(lambda row: [train_labels.columns[i] for i in range(len(row)) if row[i] == 1],
                                      axis=1)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise(f"USAGE: {sys.argv[0]} <train.feats> <train.labels.0> <train.labels.1> <test.feats>")
    train_feats = sys.argv[1]
    train_labels_zero = sys.argv[2]
    train_labels_one = sys.argv[3]
    test_feats = sys.argv[4]

    train_df, train_labels0, train_labels1, test_df = data_load(train_feats, train_labels_zero, train_labels_one, test_feats)

    # task 1
    y_pred = pd.DataFrame(predict_metastasis_location(train_df, train_labels0, test_df),
                          columns=['אבחנה-Location of distal metastases'])
    y_pred.to_csv('part1/predictions.csv', index=False)

    # task 2
    prediction = predict_tumor_size(train_df, train_labels1, test_df)
    pd.DataFrame(prediction, columns=["אבחנה-Tumor size"]).to_csv("part2/predictions.csv", index=False)


