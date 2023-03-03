__author__ = 'dk'
import torch.nn as nn
import torch as th

class CNN_block(nn.Module):
            def __init__(self, kernel_size=25, filter_num=256):
                super(CNN_block,  self).__init__()
                self.kernel_size = kernel_size
                self.filter_num = filter_num

                self._1conv1d = nn.Conv1d(stride=1,
                                          kernel_size=kernel_size,
                                          in_channels=1,
                                          out_channels=16,
                                          padding=kernel_size//2)
                self._2maxpooling = nn.MaxPool1d(kernel_size=3, stride= 1)

                self._3conv1d = nn.Conv1d(stride=1,
                                          kernel_size=kernel_size,
                                          in_channels=16,
                                          out_channels=32,
                                          padding=kernel_size//2)
                self._4maxpooling = nn.MaxPool1d(kernel_size=kernel_size, stride= 4)
                self._6flattern = nn.Flatten()
                self._5fc = nn.Linear(in_features=144, out_features=128)

            def forward(self, x):
                x = x.unsqueeze(1)

                x = self._1conv1d(x)
                x = self._2maxpooling(x)
                x = self._3conv1d(x)
                x = self._4maxpooling(x)
                x = self._5fc(x)
                x = self._6flattern(x)

                #print('cnn x shape',x.shape)
                return x

class MIMETICModel(nn.Module):
    def __init__(self, payload_sz, packet_nb, class_nb,  gru_layer_nb=2):
        super(MIMETICModel, self).__init__()
        self.payload_sz = payload_sz
        self.packet_nb = packet_nb

        self.gru_encoder = nn.GRU(  input_size = 3,  ##包长序列、到达时间序列、方向序列、 window-size
                                    hidden_size= 64,
                                    num_layers=gru_layer_nb,
                                    bidirectional=True, batch_first=True)

        self.cnn_encoder = CNN_block()

        self.fc = nn.Linear(in_features=12288, out_features= class_nb)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, field, payload):
        batch_size= field.shape[0]

        #print('packet embed shape', packet_embed.shape)
        packet_vector, hidden = self.gru_encoder(field)
        #print('packet vector shape',packet_vector.shape)
        packet_vector = packet_vector.reshape(batch_size, -1)

        payload_vector = self.cnn_encoder(payload)

        representation = th.cat((packet_vector, payload_vector), dim=1)
        representation = self.dropout(self.fc(representation))
        return representation