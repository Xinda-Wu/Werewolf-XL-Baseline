import os
import numpy as np
from time import time
import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import label_binarize
from multiprocessing.pool import Pool
import logging
from numpy import *
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./log/631_Vgg_Classificaiton_Latest_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Cross Validation k =10 ; i = CV
def svm(i):
    train_x = pd.read_csv(f'./CV_Features_631/ClassificationFeatures/Train_CV_{i}.csv').iloc[:, 9:]
    train_y = pd.read_csv(f'./CV_Features_631/ClassificationFeatures/Train_CV_{i}.csv').iloc[:, 4]
    validation_x = pd.read_csv(f'./CV_Features_631/ClassificationFeatures/Validation_CV_{i}.csv').iloc[:, 9:]
    validation_y = pd.read_csv(f'./CV_FeCV_Features_631atures/ClassificationFeatures/Validation_CV_{i}.csv').iloc[:, 4]
    test_x = pd.read_csv(f'./CV_Features_631/ClassificationFeatures/Test_CV_{i}.csv').iloc[:, 9:]
    test_y = pd.read_csv(f'./CV_Features_631/ClassificationFeatures/Test_CV_{i}.csv').iloc[:, 4]

    encoder = LabelEncoder().fit(train_y)  # #训练LabelEncoder, 把y_train中的类别编码为0，1，2，3，4，5
    y = encoder.transform(train_y)
    y_train = pd.DataFrame(encoder.transform(train_y))  # 使用训练好的LabelEncoder对源数据进行编码
    y_valid = pd.DataFrame(encoder.transform(validation_y))
    y_test = pd.DataFrame(encoder.transform(test_y))

    # 标签降维度
    y_train = y_train.iloc[:, 0].ravel()
    y_valid = y_valid.iloc[:, 0].ravel()
    y_test = y_test.iloc[:, 0].ravel()

    # X标准化
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(train_x)
    x_valid_std = scaler.fit_transform(validation_x)
    x_test_std = scaler.fit_transform(test_x)

    # ------------
    # Gamma
    # ------------
    accuracy_list_valid, f1_list_valid, auc_list_valid = [], [], []
    gamma_range = np.logspace(-10, 1, 10, base=2)
    logger.info(gamma_range)
    for idx, gamma in enumerate(tqdm(gamma_range)):
        # ------------
        # Training
        # ------------
        time0 = time()
        logger.info(f">>>>>>>CV = {i}/10, Start Trainng {idx + 1}/{len(gamma_range)}>>>>>>>")
        print(f">>>>>>> CV = {i}/10, Start Training {idx + 1}/{len(gamma_range)}>>>>>>>")
        clf = OneVsRestClassifier(
            SVC(kernel='rbf',  #
                gamma=gamma,
                C=1,  # default
                degree=1,
                cache_size=5000,
                probability=True,
                class_weight='balanced'))
        clf.fit(x_train_std, y_train)
        # ------------
        # Validation: Fine-tuning on Validation dataset
        # ------------
        y_prediction_valid = clf.predict(x_valid_std)
        accuracy_valid = accuracy_score(y_valid, y_prediction_valid)
        accuracy_list_valid.append(accuracy_valid)
        f1_valid = f1_score(y_valid, y_prediction_valid, average="weighted")
        f1_list_valid.append(f1_valid)
        y_binary_valid = label_binarize(y_valid, classes=list(range(6)))
        result_valid = clf.decision_function(x_valid_std)
        auc_valid = roc_auc_score(y_binary_valid, result_valid, average='micro')
        auc_list_valid.append(auc_valid)
        # Logger
        logger.info(f"Validation Gamma >>> Acc. = {accuracy_valid}, F1-Score = {f1_valid}, AUC = {auc_valid}")
        print(f"Validation Gamma >>> Acc. = {accuracy_valid}, F1-Score = {f1_valid}, AUC = {auc_valid}")
        print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

    best_gamma = gamma_range[accuracy_list_valid.index(max(accuracy_list_valid))]
    best_acc = max(accuracy_list_valid)
    best_f1 = f1_list_valid[accuracy_list_valid.index(max(accuracy_list_valid))]
    best_auc = auc_list_valid[accuracy_list_valid.index(max(accuracy_list_valid))]
    print(
        f"Validation >>> Best gamma = {best_gamma}, Acc. ={best_acc}, F1-Score = {best_f1}, AUC = {best_auc}\n")
    logger.info(
        f"Validation >>> Best gamma = {best_gamma}, Acc. ={best_acc}, F1-Score = {best_f1}, AUC = {best_auc}")

    # ------------
    # C
    # ------------
    best_gamma = gamma_range[accuracy_list_valid.index(max(accuracy_list_valid))]
    C = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    accuracy_list_C_valid = []
    for idx, c in enumerate(tqdm(C)):
        time0 = time()
        logger.info(f">>>>>>>CV = {i}/10, Fine-Tuining C, Start Trainng {idx + 1}/{len(C)}>>>>>>>")
        print(f">>>>>>> CV = {i}/10, Fine-Tuining C, Start Training {idx + 1}/{len(C)}>>>>>>>")
        clf = OneVsRestClassifier(
            SVC(kernel='rbf',  #
                gamma=best_gamma,
                C=c,  # default
                degree=1,
                cache_size=5000,
                probability=True,
                class_weight='balanced'))
        clf.fit(x_train_std, y_train)
        # ------------
        # Validation: Fine-tuning on Validation dataset
        # ------------
        y_prediction_valid = clf.predict(x_valid_std)
        accuracy_valid = accuracy_score(y_valid, y_prediction_valid)
        accuracy_list_C_valid.append(accuracy_valid)
        f1_valid = f1_score(y_valid, y_prediction_valid, average="weighted")
        y_binary_valid = label_binarize(y_valid, classes=list(range(6)))
        result_valid = clf.decision_function(x_valid_std)
        auc_valid = roc_auc_score(y_binary_valid, result_valid, average='micro')
        # Logger
        logger.info(f"Validation C >>> Acc. = {accuracy_valid}, F1-Score = {f1_valid}, AUC = {auc_valid}")
        print(f"Validation C >>> Acc. = {accuracy_valid}, F1-Score = {f1_valid}, AUC = {auc_valid}")
        print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
    best_c = C[accuracy_list_C_valid.index(max(accuracy_list_C_valid))]

    # logger
    best_acc = max(accuracy_list_C_valid)
    best_f1 = f1_list_valid[accuracy_list_valid.index(max(accuracy_list_valid))]
    best_auc = auc_list_valid[accuracy_list_valid.index(max(accuracy_list_valid))]
    print(
        f"Validation >>> Best gamma = {best_gamma}, Best C = {best_c}, Acc. ={best_acc}, F1-Score = {best_f1}, AUC = {best_auc}\n")
    logger.info(
        f"Validation >>> Best gamma = {best_gamma}, Best C = {best_c}, Acc. ={best_acc}, F1-Score = {best_f1}, AUC = {best_auc}")

    # ------------
    # Test: Test on Test dataset with best gamma
    # ------------
    clf_best_test = OneVsRestClassifier(
        SVC(kernel='rbf',  #
            gamma=best_gamma,
            C=best_c,  # default
            degree=1,
            cache_size=5000,
            probability=True,
            class_weight='balanced'))
    clf_best_test.fit(x_train_std, y_train)
    # accuracy & F1 & AUC on Test dataset
    y_test_prediction = clf_best_test.predict(x_test_std)
    test_accuracy = round(accuracy_score(y_test, y_test_prediction), 4)
    test_f1 = round(f1_score(y_test, y_test_prediction, average="weighted"), 4)
    y_test_binary = label_binarize(y_test, classes=list(range(6)))  # 转化为one-hot
    result_test = clf_best_test.decision_function(x_test_std)
    test_auc = round(roc_auc_score(y_test_binary, result_test, average='micro'), 4)
    print(f"CV = {i}, Test >>> gamma = {best_gamma}, Acc. ={test_accuracy}, F1-Score = {test_f1}, AUC = {test_auc}")
    logger.info(
        f"CV = {i}, Test >>> gamma = {best_gamma}, Acc. ={test_accuracy}, F1-Score = {test_f1}, AUC = {test_auc}")

    # save
    result_test = clf_best_test.predict_proba(x_test_std)
    df = pd.DataFrame(result_test)
    df.to_csv(f"./Prediction_202106_Ratio631/categorical_vggish_6pnn_20210621_prediction_CV{i}_Gamma_{round(best_gamma,4)}_C_{round(best_c)}_ACC_{test_accuracy}_F1_{test_f1}_AUC_{test_auc}.csv")
    df2 = pd.DataFrame(y_test)
    df2.to_csv(f"./Prediction_202106_Ratio631/categorical_vggish_6pnn_20210324_GT_CV{i}.csv")
    print(f">>>>>>> CV = {i}/10, Over Training >>>>>>>\n")
    logger.info(f">>>>>>> CV = {i}/10,Over Training >>>>>>>")
    return [test_accuracy, test_f1, test_auc]


if __name__ == '__main__':
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(func=svm, args=[i]) for i in range(1, 11)]
    pool.close() # 关闭pool，使其不在接受新的（主进程）任务
    average_acc_test, average_f1_test, average_auc_test = [], [], []
    for item in futures:
        result = item.get()
        average_acc_test.append(result[0])
        average_f1_test.append(result[1])
        average_auc_test.append(result[2])
    print(f"Vggish Classification Average Results: Acc.= {mean(average_acc_test)}, F1 = {mean(average_f1_test)}, AUC = {mean(average_auc_test)}")
    print(
        f"average_acc_test = {average_acc_test},/n average_f1_test={average_f1_test},/n average_auc_test = {average_auc_test}")
    logger.info(
        f"Vggish Classification Average Results: Acc.= {mean(average_acc_test)}, F1 = {mean(average_f1_test)}, AUC = {mean(average_auc_test)}")
    pool.join()


