
# "sudo python3 -m torch.distributed.launch PNN_LSTM_3.py"
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import PNN_FileDatabase
import PNN_TrainFileDataset
from VGGFace_conv6 import VGG_Face
from torch.utils.tensorboard import SummaryWriter
from LSTM_conv6_PNN import FineTuneLstmModel
import random
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score, auc,roc_curve
from sklearn.preprocessing import label_binarize
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP

# Writer will output to ./runs/ directory by default
writer_train = SummaryWriter(log_dir='./runs_lstm3/PNN_train_paper_3/')
writer_test = SummaryWriter(log_dir='./runs_lstm3/PNN_test_paper_3')

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
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# start_epoch
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
args = parser.parse_known_args()[0]

classes = ('中性', '高兴', '疑惑', '紧张',
           '轻松', '尴尬')

best_prec1 = 0

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
    train_correct = 0
    train_total = 0
    for i, videoFrames in enumerate(tqdm(train_loader)):
        # print(videoFrames)
        label = videoFrames['label'].to(device)
        # print("label = ", label)
        # [11,16,3,244,244] --> [11*16,3,244,244]
        # print("videoFrames['videoFrames'] = ",videoFrames['videoFrames'].shape)
        videoFrames = videoFrames['videoFrames'].reshape([-1, 3, 224, 224]).to(device)
        # print("videoFrames = ", videoFrames.shape)
        # [11 * 16, 3, 244, 244]--> [11 * 16,2622]
        features = featureExtractor(videoFrames.float())
        # print("features.shape =", features.shape)
        # [1, 11 * 16,2622]
        #  torch.Size([128, 6])
        output, hidden = model(features.unsqueeze(0)) # input(seq_len, batch, input_size)
        # print("output.shape=", output.shape ) # output(seq_len, batch, hidden_size * num_directions)
        #  torch.Size([8, 16, 6])
        output = output.reshape([-1, 16, 6])
        # [8, 6]
        output_mean = torch.mean(output, dim=1)
        # print("output_mean.shape=", output_mean.shape)
        loss = criterion(output_mean, label)
        # print("loss", loss)
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

        _, predicted = torch.max(output_mean.data, 1)
        # print("predicted = ", predicted)
        # print("label=",label)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()


    train_accuracy = 100 * train_correct / train_total
    print('Accuracy of the network on the Training images: %d %%' % (train_accuracy))
    writer_train.add_scalar("MSELoss_PNN", loss_all, epoch)
    writer_train.add_scalar("Accuracy", train_accuracy, epoch)


def test(test_loader, featureExtractor, model, criterion, optimizer, epoch, device):
    print(" start to test! ")
    model.eval()  # switch to train
    losses = 0.0
    test_correct = 0
    test_total = 0
    test_accuracy = 0.0
    with torch.no_grad():
        groundTruth = []
        prediction_max = []
        prediction_prob = []
        test_correct = 0
        test_total = 0
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
                output, hidden = model(features.unsqueeze(0))
                output_mean = torch.mean(output, dim=0)  # one serie of frames = 16
                Outputs.append(output_mean.data.cpu().numpy().tolist())  # All series of frames

            Outputs = torch.Tensor(Outputs)

            if Outputs.shape[0]>1:
                outputs_average = torch.mean(Outputs, dim=0).unsqueeze(0)  # average of All series' output
            # =================
            # ground truth
            groundTruth.append(label.item())
            # prediction 预测
            _, predicted = torch.max(outputs_average.data, 1)
            prediction_max.append(predicted.item())
            # 预测的概率
            prediction_prob_b = F.softmax(outputs_average.data)
            prediction_prob.append(prediction_prob_b.data.numpy().reshape(6).tolist())


            # loss 累积
            loss = criterion(outputs_average.to(device), label)
            if torch.isnan(loss):
                continue
            else:
                losses += loss.item()
                print("losses = ", losses)

            test_total += label.size(0)
            test_correct += (predicted == label.data.cpu()).sum().item()


            # Print
            if i % 200 == 0:
                print(" Test: Epoch = {0}/{1}, VideoRegression ={2}/{3}, losses = {4}".format(
                    epoch, args.epochs, i, len(test_loader), losses))
                print("ground truth.length =", len(groundTruth))
                print("prediction.length =",len(prediction_max))

                accuracy = accuracy_score(prediction_max, groundTruth)
                print("accuracy = ", accuracy)
                f1 = f1_score(prediction_max, groundTruth, average="weighted")
                print("f1", f1)
                label = label_binarize(groundTruth, classes=list(range(6)))
                print("AUC = ", roc_auc_score(label, prediction_prob, average='micro'))

                test_accuracy = 100 * test_correct / test_total

                print('Accuracy of the network on the Test images: %d' % (test_accuracy))

        accuracy = accuracy_score(prediction_max, groundTruth)
        print("accuracy = ", accuracy)
        f1 = f1_score(prediction_max, groundTruth, average="weighted")
        print("prediction_max", f1)
        label = label_binarize(groundTruth, classes=list(range(6)))
        AUC_value = roc_auc_score(label, prediction_prob, average='micro')
        print("AUC = ", AUC_value)

        # 与train 的loss大小规模相近（可以在同一张图中显示）
        test_accuracy = 100 * test_correct / test_total
        print("MSRLOSS_Test_PNN=", losses)
        print('Accuracy of the network on the Test images: %d %%' % (test_accuracy))
        writer_test.add_scalar("MSELoss_PNN", losses, epoch)
        writer_test.add_scalar('Accuracy', test_accuracy, epoch)
        writer_test.add_scalar('F1-Score', f1, epoch)
        writer_test.add_scalar('AUC', AUC_value, epoch)
        # save pro
        pro = np.array(prediction_prob)
        df2 = pd.DataFrame(pro)
        df2.to_csv("categorical04_lstm_6pnn_"+str(epoch)+"_"+str(test_accuracy)+".csv")
        return test_accuracy


def main():
    global args, start_loss, distance, best_prec1
    set_seed()
    args = parser.parse_args()

    # Load Data
    train_dataset = PNN_TrainFileDataset.FlameSet()  # for training 0.7
    test_dataset = PNN_FileDatabase.FlameSet()  # for test  0.3

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=False)  # pin_memory: 如果True，数据加载器会在返回之前将Tensors复制到CUDA固定内存

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=False)

    # Multiple GPU training , load device
    os.environ["CUDA_VISIBLE_DEVICES"] = '8,9'  # 想使用的GPU编号
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

    torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23467', rank=0, world_size=1)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    # loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)  # 并行化损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training on epochs
    for epoch in range(args.start_epochs, args.epochs):
    # for epoch in range(0, 1):
        # ===============  Saving Model  ===================
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.module.state_dict(), './PNN_saved_model_2/PNN_LSTM_' + str(epoch) + ".pth")

        # ===============  Training Model  ===================
        train(train_loader, featureExtractor, model, criterion, optimizer, epoch, device)

        # ===============  Testing Model  ===================
        if epoch % 5==0 :
            # 每5轮测试一下
            accuracy = test(test_loader, featureExtractor, model, criterion, optimizer, epoch, device)
            # ===============  Saving Model  ===================
            is_best = accuracy > best_prec1
            best_prec1 = max(accuracy, best_prec1)
            if is_best:
                torch.save(model.module.state_dict(), './PNN_saved_model_2/PNN_LSTM_' + str(epoch) +'_'+str("%.4f" % best_prec1) + ".pth")

                print("save the VGG_LSTM model  {0} model".format(best_prec1))

        print('Finished Training')

if __name__ == '__main__':
    main()
