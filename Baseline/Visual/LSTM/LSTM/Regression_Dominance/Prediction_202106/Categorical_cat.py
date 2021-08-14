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

file1 = 'eval_dominance_CV_1_RMSE_0.536606884850737_Spearman_0.21159601092895816_CCC_0.046026107828830876.csv'
file2 = 'eval_dominance_CV_2_RMSE_0.47162978121717253_Spearman_-0.006207741927128972_CCC_0.02128040605321957.csv'
file3 = 'eval_dominance_CV_3_RMSE_0.3516298631122167_Spearman_0.13196829539217883_CCC_0.056169176337182436.csv'
file4 = 'eval_dominance_CV_4_RMSE_0.34709312317590474_Spearman_0.13237795999785262_CCC_0.12009735393146626.csv'
file5 = 'eval_dominance_CV_5_RMSE_0.3474418033159782_Spearman_0.15201076484054707_CCC_0.08339709967562958.csv'
file6 = 'eval_dominance_CV_6_RMSE_0.3569488317637495_Spearman_0.1373144187193684_CCC_0.05214347240655559.csv'
file7 = 'eval_dominance_CV_7_RMSE_0.29223193737947795_Spearman_0.18007448188591826_CCC_0.13393572295067346.csv'
file8 = 'eval_dominance_CV_8_RMSE_0.2744766906235569_Spearman_0.29879520053764935_CCC_0.1869550602952993.csv'
file9 = 'eval_dominance_CV_9_RMSE_0.33949854673203056_Spearman_0.037037595462260084_CCC_0.019095022023943975.csv'
file10 = 'eval_dominance_CV_10_RMSE_0.33095986382158704_Spearman_0.1554904843534075_CCC_0.10306061342033845.csv'

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
df.to_csv('./final_decision/LSTM_Regression_D_prediction_result.csv')
