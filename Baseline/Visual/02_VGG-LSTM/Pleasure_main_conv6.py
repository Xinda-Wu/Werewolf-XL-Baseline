"""
吴鑫达模型优化：
1。 lossMIN调整
2。 tensorboard一起优化
sudo python3 -m torch.distributed.launch Pleasure_main_conv6.py

1. 特征提取 从 layer-6上面： 4096
2. 查看loss 为什么都是 0




"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import Pleasure_FileDatabase
import Pleasure_TrainFileDataset
from VGGFace_conv6 import VGG_Face
from torch.utils.tensorboard import SummaryWriter
from LSTM_conv6 import FineTuneLstmModel
import random
from torch.nn.parallel import DistributedDataParallel as DDP

# Writer will output to ./runs/ directory by default
writer_train = SummaryWriter(log_dir='./runs/Pleasure_train/')
writer_test = SummaryWriter(log_dir='./runs/Pleasure_test/')

parser = argparse.ArgumentParser(description='Training')  # description: 描述这个参数解析器是用于干什么的
# 当'-'和'--'同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分
parser.add_argument('--local_rank', type=int, )
# batch_size
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
# test_batch_size
parser.add_argument('-tb', '--test_batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
# workers
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
# epochs
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# start_epoch
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
args = parser.parse_known_args()[0]


def set_seed(seed=1):  # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_GPUS(model, model_path):
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def load_checkpoint(model, path, optimizer):
    if os.listdir(path) != None:
        model_CKPT = torch.load(path)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loaded checkpoint')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer


def train(train_loader, featureExtractor, model, criterion, optimizer, epoch, device):
    print(" start to train! ")
    model.train()  # switch to train

    loss_all = 0.0
    for i, videoFrames in enumerate(tqdm(train_loader)):
        label = videoFrames['label'].to(device)
        # print("label = ", label)
        # [11,16,3,244,244] --> [11*16,3,244,244]
        videoFrames = videoFrames['videoFrames'].reshape([-1, 3, 224, 224]).to(device)
        # [11 * 16, 3, 244, 244]--> [11 * 16,2622]
        features = featureExtractor(videoFrames.float())
        # [1, 11 * 16,2622]
        output, hidden = model(features.unsqueeze(0))
        # []
        output = output.reshape([-1, 16, 1])
        output = output.reshape([-1, 16])
        output_mean = torch.mean(output, dim=1)
        loss = criterion(output_mean, label)
        loss_all += loss.item()
        # zero the parameter gradients
        optimizer.zero_grad()
        # compute gradient
        loss.backward()
        optimizer.step()
        # Print
        if i % 200 == 0:
            print(" Training: Epoch = {0}/{1}, VideoRegression ={2}/{3}, losses = {4}".format(
                epoch, args.epochs, i, len(train_loader), loss_all))

    writer_train.add_scalar("MSELoss", loss_all, epoch)


def test(test_loader, featureExtractor, model, criterion, optimizer, epoch, device):
    print(" start to test! ")
    model.eval()  # switch to train
    losses = 0.0
    with torch.no_grad():
        for i, videoFrames in enumerate(tqdm(test_loader)):
            label = videoFrames['label'].to(device)
            videoFrames = torch.squeeze(videoFrames['videoFrames']).to(device)
            length = videoFrames.shape[0]
            Outputs = []

            if length < 16:
                lack = 16 - length
                repeat_frames = videoFrames[-1:, ...].repeat(lack, 1, 1, 1)
                videoFrames = torch.cat((videoFrames, repeat_frames), 0)

            circle = int(length / 8) - 1
            for k in range(circle):
                start = 0 + 8 * k
                end = 16 + 8 * k
                features = featureExtractor(videoFrames[start:end, ...].float())
                # print("input from {0} to {1}, now/circle = {2}/{3}".format(start, end, k, circle))
                # print(features.shape)
                output, hidden = model(features.unsqueeze(0))
                # input_size =1   Input[bacht_size, num_class=1 for regression]
                output_mean = torch.mean(output).unsqueeze(0)  # one serie of frames = 16
                # print("output_mean = ", output_mean)
                Outputs.append(output_mean)  # All series of frames

            Outputs = torch.Tensor(Outputs)
            outputs_average = torch.mean(Outputs)  # average of All series' output
            loss = criterion(outputs_average.to(device), label)

            if torch.isnan(loss):
                continue
            else:
                losses += loss.item()

            # Print
            if i % 200 == 0:
                print(" Test: Epoch = {0}/{1}, VideoRegression ={2}/{3}, losses = {4}".format(
                    epoch, args.epochs, i, len(test_loader), losses))

        # 与train 的loss大小规模相近（可以在同一张图中显示）
        writer_test.add_scalar("MSELoss", losses, epoch)
        print("MSRLOSS_Test=",losses)
        return losses


def main():
    global args, start_loss, distance
    set_seed()
    args = parser.parse_args()

    # Load Data
    train_dataset = Pleasure_TrainFileDataset.FlameSet()  # for training 0.7
    test_dataset = Pleasure_FileDatabase.FlameSet()  # for test  0.3

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=False)  # pin_memory: 如果True，数据加载器会在返回之前将Tensors复制到CUDA固定内存

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=False)

    # Multiple GPU training , load device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # 想使用的GPU编号
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU开始编号，依次往下
    print("load into device:", device)

    # init visual feature extractor
    featureExtractor = VGG_Face()
    featureExtractor = load_GPUS(featureExtractor, './vgg16_face_30.pth')
    featureExtractor = nn.DataParallel(featureExtractor)
    featureExtractor.to(device)
    print('load visual feature extractor success')

    # load LSTM model, 多GPU计算
    model = FineTuneLstmModel()
    # model = model.load_state_dict(torch.load('./VGG_LSTM_4.pth'))
    # print("load pretrain model success!")
    # 使用单机多卡
    # sudo python3 -m torch.distributed.launch main.py
    # 注: 这里如果使用了argparse, 一定要在参数里面加上--local_rank
    torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23462', rank=0, world_size=1)
    # 先将model加载到GPU, 然后才能使用DistributedDataParallel进行分发
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    # loss criterion and optimizer
    criterion = nn.MSELoss()
    criterion = criterion.to(device)  # 并行化损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,)

    # Training on epochs
    for epoch in range(args.start_epochs, args.epochs):

        # train & test
        train(train_loader, featureExtractor, model, criterion, optimizer, epoch, device)
        loss = test(test_loader, featureExtractor, model, criterion, optimizer, epoch, device)

        # ===============  Saving Model  ===================
        if epoch % 5 != 0:
            torch.save(model, './saved_model_Pleasure/lstm_loss_' + str("%.4f" % loss) + '.pth')
            torch.save({'epoch': epoch,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'test_best_loss': loss, },
                       './saved_model_Pleasure/lstm_loss_' + str("%.4f" % loss) + '.pth.tar')
            print("save the min loss {0} model".format(loss))


if __name__ == '__main__':
    main()
