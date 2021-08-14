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

filepath ='Speaker_Visual_hog_final.csv'
data = pd.read_csv(filepath)
x = data.iloc[:, 6:]  # 数据特征
y = data.iloc[:,2]  # 标签

# 将数据划分为训练集和测试集，test_size=.3表示30%的测试集, 随机数种子, 保证可复现性
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=324)

# 修正测试集和训练集的索引
for i in [x_train, x_test, y_train, y_test ]:
    i.index  = range(i.shape[0])

# 标准化
scaler_x = StandardScaler()
# scaler_y = StandardScaler()
x_train_std = scaler_x.fit_transform(x_train)
x_test_std = scaler_x.fit_transform(x_test)

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

# Spend Time
time0 = time()

# Basic SVM Model (gamma =  0.04888888888888889,  C = 1.13333333333,)
# sklearn通过OneVsRestClassifier实现svm.SVC的多分类
clf = SVR(kernel = 'rbf', cache_size=5000)

# 超参数 Gamma
gamma_range = np.logspace(-10, 1, 10, base=2) # 返回13个数字，底是2
print(gamma_range)

parameters = {
"C": [1],
"kernel": ["rbf"],
"degree":[1],
"gamma":gamma_range,
}

# evaluation Metrics
score = 'neg_mean_squared_error'

# Grid Search params
model_tunning = GridSearchCV(clf,
                             param_grid=parameters,
                             n_jobs=-1,
                             cv=5,
                             verbose = 32,
                             scoring=score)
model_tunning.fit(x_train_std, y_train)

# # # # # # # # #
# predictions
# # # # # # # # #
results = {}
bst = model_tunning.best_estimator_
result = bst.predict(x_test_std)
# accuracy
rmse = sqrt(mean_squared_error(y_test, result))
results["RMSE"] = rmse
print("(1) Evaluation - RMSE = ", rmse)

# Spearman
data = {'result':result, 'y_test':y_test}
df = pd.DataFrame(data, columns=['result','y_test'])
spearman = df.corr(method="spearman" )
print("(2) Evaluation - Spearmman = \n", spearman)

# CCC
prediction = result
ground = y_test
meanfe = getMean(ground)
meanexp = getMean(prediction)
meanofdiff = getMeanofDiffs(ground,prediction)
pvarfe = getPvar(ground, meanfe)
pvarexp = getPvar(prediction, meanexp)
ccc = getCCC(pvarfe,pvarexp,meanofdiff,meanfe,meanexp)
print('(3) Evaluation - CCC =  ' + str(ccc))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
print()
print()

# results
df = pd.DataFrame(data={"hog_prediction_p": prediction, "hog_groundtruth_p": y_test.values.tolist()})
df.to_csv("/mnt/nfs-shared/xinda/Werewolf-XL/werewolf-XL_202103/SVM&SVR/visuail/hog/results/hog_pleasure.csv")
print("save success!")