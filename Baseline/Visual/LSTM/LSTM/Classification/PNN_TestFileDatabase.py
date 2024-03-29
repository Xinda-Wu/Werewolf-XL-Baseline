import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd


classes = ('中性','尴尬' , '疑惑', '紧张',
           '轻松', '高兴')
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
        csvPath = f'/mnt/nfs-shared/xinda/Werewolf-XL/Werewolf-XL_202106/2_LSTM/LSTM/Split_dataset/CV_Features/ClassificationFeatures/Test_CV_{cv}.csv'
        df = pd.read_csv(csvPath)
        pre_fillename = df['VideoName'].values
        # Pleasure_average,Arousal_average,Dominance_average
        labels = df['Categorical'].values
        self.videoFrame=pre_fillename
        self.transforms=transform
        self.labels = labels

    def __getitem__(self, index):

        # label
        label =self.labels[index]
        label_index = classes.index(label)

        # 文件夹
        allFrames = self.videoFrame[index]
        # print("单个图片文件夹：", allFrames)
        # 所有图片的地址
        imgsFilePath = os.path.join('/mnt/nfs-shared/xinda/Werewolf-XL/werewolf_video/features', allFrames+"_aligned")
        imgs = os.listdir(imgsFilePath)
        imgsPath = [os.path.join(imgsFilePath,k) for k in imgs]
        imgsPath.sort()

        # 加载数据
        returnFrames = torch.from_numpy(np.zeros((len(imgs), 3, 224, 224)))
        # print("Video = {0},  The number of frames = {1} ".format(allFrames, returnFrames.shape))

        for i in range(len(imgs)):
            pil_img = Image.open(imgsPath[i])
            if self.transforms:
                data = self.transforms(pil_img)
            else:
                pil_img = np.asarray(pil_img)
                data = torch.from_numpy(pil_img)
            returnFrames[i] = data
        dict = {"videoFrames":returnFrames, "label": label_index}
        return dict

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    dataSet=FlameSet()
    print(dataSet[0])

