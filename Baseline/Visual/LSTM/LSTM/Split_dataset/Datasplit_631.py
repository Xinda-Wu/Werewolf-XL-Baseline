import os
import numpy as np
import pandas as pd

all_session = [i for i in range(1, 31)]
print(all_session)
cv_dataset = {}
for i in range(0, 10):  # CV = 10
    if i < 7:
        cv_dataset[f"CV_{i + 1}"] = {
            "Test": all_session[i * 3:3 * (i + 1)],
            "Validation": all_session[(i + 1) * 3:(i + 1) * 3 + 9],
            "Train": list(
                set(all_session) - set(all_session[i * 3:3 * (i + 1)]) - set(all_session[(i + 1) * 3:(i + 1) * 3 + 6]))
        }
    if i == 7:
        cv_dataset[f"CV_{i + 1}"] = {
            "Test": all_session[i * 3:3 * (i + 1)],
            "Validation": [25, 26, 27, 28, 29, 30, 1, 2, 3],
            "Train": list(
                set(all_session) - set(all_session[i * 3:3 * (i + 1)]) - set([25, 26, 27, 28, 29, 30, 1, 2, 3]))
        }
    if i == 8:
        cv_dataset[f"CV_{i + 1}"] = {
            "Test": all_session[i * 3:3 * (i + 1)],
            "Validation": [28, 29, 30, 1, 2, 3, 4, 5, 6],
            "Train": list(set(all_session) - set(all_session[i * 3:3 * (i + 1)]) - set([28, 29, 30, 1, 2, 3, 4, 5, 6]))
        }
    if i == 9:
        cv_dataset[f"CV_{i + 1}"] = {
            "Test": all_session[i * 3:3 * (i + 1)],
            "Validation": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Train": list(set(all_session) - set(all_session[i * 3:3 * (i + 1)]) - set([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        }

for idx, item in enumerate(cv_dataset):
    print(f"CV = {idx + 1}")
    print(f"Test >>> {cv_dataset[item]['Test']} Validation >>> {cv_dataset[item]['Validation']}")
    print(f"Train >>> {cv_dataset[item]['Train']}\n")

# --------------------------------
# split  classification dataset
# --------------------------------

# manually delete the first two cols
features = pd.read_csv('./1_Features_raw/Classification_Speaker_6pnn_202106_session2.csv')
participate = pd.read_csv("Final_results_0620_1721.csv")
for cv_idx, key in enumerate(cv_dataset):
    Test_sessions = cv_dataset[key]["Test"]
    print("Test sessions = ", Test_sessions)
    Validation_sessions = cv_dataset[key]["Validation"]
    Train_sessions = cv_dataset[key]["Train"]

    # ---------------------
    # load data
    # ---------------------
    Test_data = features[features['Session'].isin(Test_sessions)]
    Test_data.to_csv(f"./CV_Features_631/ClassificationFeatures/Test_CV_{cv_idx + 1}.csv")
    Validation_data = features[features['Session'].isin(Validation_sessions)]
    Train_data = features[features['Session'].isin(Train_sessions)]

    # ---------------------
    # speaker independent
    # ---------------------
    # (1) delete instance
    delete_info = set()
    Test_Overlap = participate[participate['Round'].isin(cv_dataset[key]["Test"])]
    for idx, item in Test_Overlap.iterrows():
        values = set(item.values[6:-1])
        for temp in values:
            if type(temp) != float:
                if int(temp.split('-')[0]) not in Test_sessions:
                    if int(temp.split('-')[0]) < 10:
                        temp = '0' + temp
                        delete_info.add(temp.replace("-", "_"))
                    else:
                        delete_info.add(temp.replace("-", "_"))
    print(delete_info)

    # (2) 整理Validation and Train data
    for idx, item in Validation_data.iterrows():
        if item['VideoName'][:5] in list(delete_info):
            Validation_data.drop(labels=idx, inplace=True)
    Validation_data.to_csv(f"./CV_Features_631/ClassificationFeatures/Validation_CV_{cv_idx + 1}.csv")

    for idx, item in Train_data.iterrows():
        if item['VideoName'][:5] in list(delete_info):
            Train_data.drop(labels=idx, inplace=True)
    Train_data.to_csv(f"./CV_Features_631/ClassificationFeatures/Train_CV_{cv_idx + 1}.csv")


# --------------------------------
# split  Regression dataset
# --------------------------------
# manually delete the first two cols
features = pd.read_csv('./1_Features_raw/Regression_Speaker_202106_session2.csv')
participate = pd.read_csv("Final_results_0620_1721.csv")
for cv_idx, key in enumerate(cv_dataset):
    Test_sessions = cv_dataset[key]["Test"]
    print("Test sessions = ", Test_sessions)
    Validation_sessions = cv_dataset[key]["Validation"]
    Train_sessions = cv_dataset[key]["Train"]

    # ---------------------
    # load data
    # ---------------------
    Test_data = features[features['Session'].isin(Test_sessions)]
    Test_data.to_csv(f"./CV_Features_631/RegressionFeatures/Test_CV_{cv_idx + 1}.csv")
    Validation_data = features[features['Session'].isin(Validation_sessions)]
    Train_data = features[features['Session'].isin(Train_sessions)]

    # ---------------------
    # speaker independent
    # ---------------------
    # (1) delete instance
    delete_info = set()
    Test_Overlap = participate[participate['Round'].isin(cv_dataset[key]["Test"])]
    for idx, item in Test_Overlap.iterrows():
        values = set(item.values[6:-1])
        for temp in values:
            if type(temp) != float:
                if int(temp.split('-')[0]) not in Test_sessions:
                    if int(temp.split('-')[0]) < 10:
                        temp = '0' + temp
                        delete_info.add(temp.replace("-", "_"))
                    else:
                        delete_info.add(temp.replace("-", "_"))
    print(delete_info)

    # (2) 整理Validation and Train data
    for idx, item in Validation_data.iterrows():
        if item['VideoName'][:5] in list(delete_info):
            Validation_data.drop(labels=idx, inplace=True)
    Validation_data.to_csv(f"./CV_Features_631/RegressionFeatures/Validation_CV_{cv_idx + 1}.csv")

    for idx, item in Train_data.iterrows():
        if item['VideoName'][:5] in list(delete_info):
            Train_data.drop(labels=idx, inplace=True)
    Train_data.to_csv(f"./CV_Features_631/RegressionFeatures/Train_CV_{cv_idx + 1}.csv")