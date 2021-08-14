import numpy as np
import os


files = os.listdir('./')
rmses = []
spearmans =[]
cccs = []
for idx, item in enumerate(files):
    # print(item)
    if 'eval' in item:
        items = item.split('_')
        rmse = items[5]
        rmses.append(float(rmse))
        spearman = items[7]
        spearmans.append(float(spearman))
        ccc = items[9][:-4]
        cccs.append(float((ccc)))

print(np.mean(rmses))
print(np.mean(spearmans))
print(np.mean(cccs))