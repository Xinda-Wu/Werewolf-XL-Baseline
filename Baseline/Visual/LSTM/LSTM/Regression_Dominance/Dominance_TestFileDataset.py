import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import random

# 数据处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

#定义自己的数据集合
class FlameSet_Test(data.Dataset):
    def __init__(self, cv):
        # 所有图片的绝对路径
        csvPath = f'/mnt/nfs-shared/xinda/Werewolf-XL/Werewolf-XL_202106/2_LSTM/LSTM/Split_dataset/CV_Features/RegressionFeatures/Test_CV_{cv}.csv'
        df = pd.read_csv(csvPath)
        pre_fillename = df['VideoName'].values
        # Pleasure_average,Arousal_average,Dominance_average
        labels = df['Mean_D_1_2_4'].values
        self.videoName=pre_fillename
        self.transforms=transform
        self.labels = labels

    def __getitem__(self, index):

        # label
        label =self.labels[index]

        # 文件夹
        allFrames = self.videoName[index]
        # print("单个图片文件夹：", allFrames)
        # 所有图片的地址
        imgsFilePath = os.path.join('/mnt/nfs-shared/xinda/Werewolf-XL/werewolf_video/features', allFrames+"_aligned")
        imgs = os.listdir(imgsFilePath)
        imgsPath = [os.path.join(imgsFilePath,k) for k in imgs]
        imgsPath.sort()

        if len(imgsPath) >= 16:
            # 任意选择16张图片作为输入
            frameIndexList = random.sample(range(0, len(imgsPath)), 16);
            train_imgPath = [imgsPath[frameIndexList[i]] for i in range(16)]
        else:
            repeat = 16 - len(imgsPath)
            repeatFrame = imgsPath[-1]
            train_imgPath = imgsPath
            for i in range(repeat):
                train_imgPath.append(repeatFrame)

        # 加载数据
        returnFrames = torch.from_numpy(np.zeros((16, 3, 224, 224)))
        # print("Video = {0},  The number of frames = {1} ".format(allFrames, returnFrames.shape))
        for i in range(16):
            pil_img = Image.open(train_imgPath[i])
            if self.transforms:
                data = self.transforms(pil_img)
            else:
                pil_img = np.asarray(pil_img)
                data = torch.from_numpy(pil_img)
            returnFrames[i] = data

        # torch.Size([16, 3, 224, 224])
        dict = {"videoFrames": returnFrames, "label": label}
        return dict

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataSet = FlameSet()
    print("=======================")
    print(dataSet[0])
