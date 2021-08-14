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
file1 = 'eval_pleasure_CV_1_RMSE_0.29004359053136397_Spearman_0.47832049447871483_CCC_0.43270260247291503.csv'
file2 = 'eval_pleasure_CV_2_RMSE_0.25596770800569063_Spearman_0.6134642579371166_CCC_0.5465878696173172.csv'
file3 = 'eval_pleasure_CV_3_RMSE_0.2162561191161807_Spearman_0.6537078510881107_CCC_0.6398429765831295.csv'
file4 = 'eval_pleasure_CV_4_RMSE_0.24771327069804125_Spearman_0.6077806684967764_CCC_0.5950608466353984.csv'
file5 = 'eval_pleasure_CV_5_RMSE_0.24145526054420016_Spearman_0.6030365477945665_CCC_0.5194490869216597.csv'
file6 = 'eval_pleasure_CV_6_RMSE_0.2189866419768109_Spearman_0.6709692918009953_CCC_0.6500684842655716.csv'
file7 = 'eval_pleasure_CV_7_RMSE_0.23125649037544177_Spearman_0.740071818237238_CCC_0.6545520890873806.csv'
file8 = 'eval_pleasure_CV_8_RMSE_0.19290294434609206_Spearman_0.5930427586501867_CCC_0.7134700618285874.csv'
file9 = 'eval_pleasure_CV_9_RMSE_0.22479051481587412_Spearman_0.6901393521717643_CCC_0.6608514575803144.csv'
file10 = 'eval_pleasure_CV_10_RMSE_0.20104490094129232_Spearman_0.6451286862350684_CCC_0.624965540487012.csv'


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
df.to_csv('./final_decision/LSTM_Regression_P_prediction_result.csv')
