import numpy as np
import os
import pandas as pd
# file1 = ''
# file2 = ''
# file3 = ''
# file4 = ''
# file5 = ''
# file6 = ''
# file7 = ''
# file8 = ''
# file9 = ''
# file10 = ''

file1 = 'eval_arousal_CV_1_RMSE_0.3221636354358591_Spearman_0.06114523939391956_CCC_0.02969973535689674.csv'
file2 = 'eval_arousal_CV_2_RMSE_0.3844106312684173_Spearman_0.0909828218263165_CCC_0.030046437257935765.csv'
file3 = 'eval_arousal_CV_3_RMSE_0.2568641052898089_Spearman_0.11425203567561194_CCC_0.1740419866991254.csv'
file4 = 'eval_arousal_CV_4_RMSE_0.26864606608368585_Spearman_0.010748879968845534_CCC_0.016081828795419417.csv'
file5 = 'eval_arousal_CV_5_RMSE_0.22346664959385248_Spearman_0.15181768985477956_CCC_0.03353028228331323.csv'
file6 = 'eval_arousal_CV_6_RMSE_0.19368459076944078_Spearman_0.10251400560790613_CCC_0.025933457386683068.csv'
file7 = 'eval_arousal_CV_7_RMSE_0.2015533265956057_Spearman_0.011536325525911686_CCC_0.014232771645381836.csv'
file8 = 'eval_arousal_CV_8_RMSE_0.23144837029131218_Spearman_-0.0037185731381242245_CCC_0.021661430694919215.csv'
file9 = 'eval_arousal_CV_9_RMSE_0.2645733256763858_Spearman_-0.005453583367755578_CCC_-0.004893477640059096.csv'
file10 = 'eval_arousal_CV_10_RMSE_0.26070153387166717_Spearman_0.19001110121453402_CCC_0.1203915451439117.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
df5 = pd.read_csv(file5)
df6 = pd.read_csv(file6)
df7 = pd.read_csv(file7)
df8 = pd.read_csv(file8)
df9 = pd.read_csv(file9)
df10 = pd.read_csv(file10)

df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10],axis=0,ignore_index=True)
print(df.shape)
df.to_csv('./final_decision/LSTM_Regression_A_prediction_result.csv')
