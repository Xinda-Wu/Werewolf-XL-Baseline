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

file1 = 'CV_1_Epoch_629_ACC_23.37778_F1_0.37896_AUC_0.65899_Categorical_lstm_6pnn_202106_2.csv'
file2 = 'CV_2_Epoch_629_ACC_24.29487_F1_0.29547_AUC_0.64256_Categorical_lstm_6pnn_202106_2.csv'
file3 = 'CV_3_Epoch_629_ACC_26.09428_F1_0.372_AUC_0.72784_Categorical_lstm_6pnn_202106_2.csv'
file4 = 'CV_4_Epoch_629_ACC_16.09388_F1_0.27726_AUC_0.72267_Categorical_lstm_6pnn_202106_2.csv'
file5 = 'CV_5_Epoch_629_ACC_40.8707_F1_0.5317_AUC_0.71782_Categorical_lstm_6pnn_202106_2.csv'
file6 = 'CV_6_Epoch_629_ACC_31.02337_F1_0.47355_AUC_0.73651_Categorical_lstm_6pnn_202106_2.csv'
file7 = 'CV_7_Epoch_629_ACC_14.25672_F1_0.24956_AUC_0.63593_Categorical_lstm_6pnn_202106_2.csv'
file8 = 'CV_8_Epoch_629_ACC_26.91415_F1_0.42413_AUC_0.6742_Categorical_lstm_6pnn_202106_2.csv'
file9 = 'CV_9_Epoch_629_ACC_25.84856_F1_0.36306_AUC_0.56568_Categorical_lstm_6pnn_202106_2.csv'
file10 = 'CV_10_Epoch_629_ACC_17.63827_F1_0.27168_AUC_0.70553_Categorical_lstm_6pnn_202106_2.csv'


df1 = pd.read_csv(file1)
print(df1.shape)
df2 = pd.read_csv(file2)
print(df2.shape)
df3 = pd.read_csv(file3)
print(df3.shape)
df4 = pd.read_csv(file4)
print(df1.shape)
df5 = pd.read_csv(file5)
print(df1.shape)
df6 = pd.read_csv(file6)
print(df1.shape)
df7 = pd.read_csv(file7)
print(df1.shape)
df8 = pd.read_csv(file8)
print(df1.shape)
df9 = pd.read_csv(file9)
print(df1.shape)
df10 = pd.read_csv(file10)
print(df1.shape)

df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10],axis=0,ignore_index=True)
print(df.shape)
df.to_csv('./final_decision/LSTM_C_prediction_result.csv')
