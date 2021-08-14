import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable
from VGGFace import VGG_Face


class FineTuneLstmModel(nn.Module):

    def __init__(self, num_classes=1, lstm_layers=1, hidden_size=128, fc_size=6):
        super(FineTuneLstmModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size

        # LSTM
        '''
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        torch.LSTM 中 batch_size 维度默认是放在第二维度，故此参数设置可以将 batch_size 放在第一维度。
        如：input 默认是(4,1,5)，中间的 1 是 batch_size，指定batch_first=True后就是(1,4,5)。
        所以，如果你的输入数据是二维数据的话，就应该将 batch_first 设置为True;
        '''
        self.rnn = nn.LSTM(input_size = 4096,
                    hidden_size = hidden_size,
                    num_layers = lstm_layers,
                    batch_first = False)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.modelName = 'vggface_lstm'


    def init_hidden(self, num_layers, batch_size):
        return (Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).cuda(),
                Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).cuda())

    def forward(self, inputs, hidden=128, steps=0):
        '''
        inputs: sequence of images:
        torch.Size([93, 3, 224, 224]) -- > [93,2622] batch_size, input_size
        吴鑫达：[16, 2622]
        '''
        outputs, hidden = self.rnn(inputs)
        outputs = self.fc(outputs[0])
        return outputs, hidden






