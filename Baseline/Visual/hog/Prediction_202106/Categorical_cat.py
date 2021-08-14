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

# file1 = 'CV1_Hog_Arousal_0621.csv'
# file2 = 'CV2_Hog_Arousal_0621.csv'
# file3 = 'CV3_Hog_Arousal_0621.csv'
# file4 = 'CV4_Hog_Arousal_0621.csv'
# file5 = 'CV5_Hog_Arousal_0621.csv'
# file6 = 'CV6_Hog_Arousal_0621.csv'
# file7 = 'CV7_Hog_Arousal_0621.csv'
# file8 = 'CV8_Hog_Arousal_0621.csv'
# file9 = 'CV9_Hog_Arousal_0621.csv'
# file10 = 'CV10_Hog_Arousal_0621.csv'

file1 = 'CV1_hog_Dominance_0621.csv'
file2 = 'CV2_hog_Dominance_0621.csv'
file3 = 'CV3_hog_Dominance_0621.csv'
file4 = 'CV4_hog_Dominance_0621.csv'
file5 = 'CV5_hog_Dominance_0621.csv'
file6 = 'CV6_hog_Dominance_0621.csv'
file7 = 'CV7_hog_Dominance_0621.csv'
file8 = 'CV8_hog_Dominance_0621.csv'
file9 = 'CV9_hog_Dominance_0621.csv'
file10 = 'CV10_hog_Dominance_0621.csv'


# file1 = 'CV1_Vggish_pleasure_0621.csv'
# file2 = 'CV2_Vggish_pleasure_0621.csv'
# file3 = 'CV3_Vggish_pleasure_0621.csv'
# file4 = 'CV4_Vggish_pleasure_0621.csv'
# file5 = 'CV5_Vggish_pleasure_0621.csv'
# file6 = 'CV6_Vggish_pleasure_0621.csv'
# file7 = 'CV7_Vggish_pleasure_0621.csv'
# file8 = 'CV8_Vggish_pleasure_0621.csv'
# file9 = 'CV9_Vggish_pleasure_0621.csv'
# file10 = 'CV10_Vggish_pleasure_0621.csv'

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
df.to_csv('./final_decision/Regression_D_prediction_result.csv')
