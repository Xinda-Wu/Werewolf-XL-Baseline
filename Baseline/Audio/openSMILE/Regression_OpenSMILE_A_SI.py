import os
import numpy as np
from time import time
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from multiprocessing.pool import Pool
from numpy import *
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./log/631_OpenSMILE_Regression_Pleasure_Latest_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def getPvar(vals, mean):
    N = len(vals)
    su = 0
    for i in range(len(vals)):
        su = su + ((vals[i]-mean)*(vals[i]-mean))
    pvar = (1/N) * su
    return pvar


def getMean(vals):
    su = 0
    for i in range(len(vals)):
        su = su + vals[i]
    mean = su/(len(vals))
    return mean


def getMeanofDiffs(xvals, yvals):
    su = 0
    for i in range(len(xvals)):
        su = su + ((xvals[i] - yvals[i])*(xvals[i] - yvals[i]))
    meanodiffs = su/(len(xvals))
    return meanodiffs


def getCCC(pvarfe,pvarexp,meanofdiff,meanfe,meanexp):
    bottom = pvarfe + pvarexp + ((meanfe - meanexp)*(meanfe - meanexp))
    answer = 1 - (meanofdiff / bottom)
    return answer


def svr(i):
    train_x = pd.read_csv(f'./CV_Features_631/RegressionFeatures/Train_CV_{i}.csv').iloc[:, 9:]
    train_y = pd.read_csv(f'./CV_Features_631/RegressionFeatures/Train_CV_{i}.csv').iloc[:, 6]
    validation_x = pd.read_csv(f'./CV_Features_631/RegressionFeatures/Validation_CV_{i}.csv').iloc[:, 9:]
    validation_y = pd.read_csv(f'./CV_Features_631/RegressionFeatures/Validation_CV_{i}.csv').iloc[:, 6]
    test_x = pd.read_csv(f'./CV_Features_631/RegressionFeatures/Test_CV_{i}.csv').iloc[:, 9:]
    test_y = pd.read_csv(f'./CV_Features_631/RegressionFeatures/Test_CV_{i}.csv').iloc[:, 6]

    # 标准化
    scaler_x = StandardScaler()
    # scaler_y = StandardScaler()
    x_train_std = scaler_x.fit_transform(train_x)
    x_valid_std = scaler_x.fit_transform(validation_x)
    x_test_std = scaler_x.fit_transform(test_x)

    # Best Gamma
    ccc_list_valid = []
    gamma_range = np.logspace(-10, 1, 10, base=2)
    for idx, gamma in enumerate(tqdm(gamma_range)):
        time0 = time()
        logger.info(f">>>>>>>CV = {i}/10, Regression Start Trainng {idx + 1}/{len(gamma_range)}>>>>>>>")
        print(f">>>>>>> CV = {i}/10, Regression Start Training {idx + 1}/{len(gamma_range)}>>>>>>>")
        # ------------
        # Training
        # ------------
        clf = SVR(kernel='rbf', gamma=gamma, cache_size=5000)
        clf.fit(x_train_std,train_y)
        # -------------
        # Validation & Evaluation
        # --------------
        prediction_valid = clf.predict(x_valid_std)
        # RMSE
        rmse = sqrt(mean_squared_error(validation_y, prediction_valid))
        # Spearman
        data_valid = {'result_valid': prediction_valid, 'y_valid': validation_y}
        df = pd.DataFrame(data_valid, columns=['result_valid', 'y_valid'])
        spearman_valid = df.corr(method="spearman")
        spearman_values_valid = spearman_valid.iloc[0].values[1]
        # CCC
        prediction = prediction_valid
        ground =validation_y
        meanfe = getMean(ground)
        meanexp = getMean(prediction)
        meanofdiff = getMeanofDiffs(ground, prediction)
        pvarfe = getPvar(ground, meanfe)
        pvarexp = getPvar(prediction, meanexp)
        ccc = getCCC(pvarfe, pvarexp, meanofdiff, meanfe, meanexp)
        ccc_list_valid.append(ccc)
        print(f" CV={i}, Idx ={idx}, Validation Result: RMSE = {rmse}, Spearman ={spearman_values_valid}, CCC = {ccc}")
        logger.info(f" CV={i}, Idx ={idx}, Validation Result: RMSE = {rmse}, Spearman ={spearman_values_valid}, CCC = {ccc}")

    # Best C
    best_gamma = gamma_range[ccc_list_valid.index(max(ccc_list_valid))]
    C = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    ccc_list_C_valid = []
    for idx, c in enumerate(tqdm(C)):
        time0 = time()
        logger.info(f">>>>>>>CV = {i}/10,  Best Gamma = {best_gamma}, Fune-tuning C, Regression Start Training {idx + 1}/{len(C)}>>>>>>>")
        print(f">>>>>>> CV = {i}/10,  Best Gamma = {best_gamma}, Fune-tuning C, Regression Start Training {idx + 1}/{len(C)}>>>>>>>")
        clf = SVR(kernel='rbf', gamma=best_gamma, cache_size=5000, C=c)
        clf.fit(x_train_std, train_y)
        # -------------
        # Validation & Evaluation
        # --------------
        prediction_valid = clf.predict(x_valid_std)
        # RMSE
        rmse = sqrt(mean_squared_error(validation_y, prediction_valid))
        # Spearman
        data_valid = {'result_valid': prediction_valid, 'y_valid': validation_y}
        df = pd.DataFrame(data_valid, columns=['result_valid', 'y_valid'])
        spearman_valid = df.corr(method="spearman")
        spearman_values_valid = spearman_valid.iloc[0].values[1]
        # CCC
        prediction = prediction_valid
        ground = validation_y
        meanfe = getMean(ground)
        meanexp = getMean(prediction)
        meanofdiff = getMeanofDiffs(ground, prediction)
        pvarfe = getPvar(ground, meanfe)
        pvarexp = getPvar(prediction, meanexp)
        ccc = getCCC(pvarfe, pvarexp, meanofdiff, meanfe, meanexp)
        ccc_list_C_valid.append(ccc)
        print(f" CV={i}, Idx ={idx}, Validation Result: RMSE = {rmse}, Spearman ={spearman_values_valid}, CCC = {ccc}")
        logger.info(
            f" CV={i}, Idx ={idx}, Validation Result: RMSE = {rmse}, Spearman ={spearman_values_valid}, CCC = {ccc}")
    best_c = C[ccc_list_C_valid.index(max(ccc_list_C_valid))]


    # -------------
    # Test
    # --------------
    clf_best_test = SVR(kernel='rbf', gamma=best_gamma, C=best_c, cache_size=5000)
    clf_best_test.fit(x_train_std, train_y)
    y_test_prediction = clf_best_test.predict(x_test_std)
    rmse_test = sqrt(mean_squared_error(test_y, y_test_prediction))
    # Spearman
    data = {'result_test': y_test_prediction, 'y_test': test_y}
    df = pd.DataFrame(data, columns=['result_test', 'y_test'])
    spearman_test = df.corr(method="spearman")
    spearman_values_test = spearman_test.iloc[0].values[1]
    # CCC
    prediction_test = y_test_prediction
    ground_test = test_y
    meanfe = getMean(ground_test)
    meanexp = getMean(prediction_test)
    meanofdiff = getMeanofDiffs(ground_test, prediction_test)
    pvarfe = getPvar(ground_test, meanfe)
    pvarexp = getPvar(prediction_test, meanexp)
    ccc_test = getCCC(pvarfe, pvarexp, meanofdiff, meanfe, meanexp)
    print(f"CV = {i}, Test >>> gamma = {best_gamma}, C = {best_c}, RMSE. ={rmse_test}, Spearman = {spearman_test}, CCC = {ccc_test}")
    logger.info(
        f"CV = {i}, Test >>> gamma = {best_gamma}, C = {best_c}, RMSE. ={rmse_test}, Spearman = {spearman_test}, CCC = {ccc_test}")

    # Save
    df = pd.DataFrame(data={"opensmile_prediction_A": prediction_test, "opensmile_groundtruth_A": test_y.values.tolist()})
    df.to_csv(f"./Prediction_202106_Ratio631/CV{i}_opensmile_Arousal_0621.csv")
    print("save success!")
    print(f">>>>>>> CV = {i}/10, Over Training >>>>>>>\n")
    logger.info(f">>>>>>> CV = {i}/10,Over Training >>>>>>>")
    return [rmse_test,spearman_values_test,ccc_test]

if __name__ == '__main__':
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(func=svr, args=[i]) for i in range(1, 11)]
    pool.close() # 关闭pool，使其不在接受新的（主进程）任务
    average_rmse_test, average_pearson_test, average_ccc_test = [], [], []
    for item in futures:
        result = item.get()
        average_rmse_test.append(result[0])
        average_pearson_test.append(result[1])
        average_ccc_test.append(result[2])
    print(f"OpenSMILE Regression Average Results of Arousal: RMSE.= {mean(average_rmse_test)}, Spearman = {mean(average_pearson_test)}, CCC = {mean(average_ccc_test)}")
    logger.info(
        f"/n/n/n OpenSMILE Regression Average Results of Arousal: RMSE.= {mean(average_rmse_test)}, Spearman = {mean(average_pearson_test)}, CCC = {mean(average_ccc_test)}")
    pool.join()