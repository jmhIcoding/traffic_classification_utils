__author__ = 'dk'
import torch.nn as nn
import torch as th

class CNN_block(nn.Module):
            def __init__(self, embedding_dim, kernel_size=7, filter_num=256):
                super(CNN_block,  self).__init__()
                self.kernel_size = kernel_size
                self.filter_num = filter_num

                self._1conv1d = nn.Conv1d(stride=1,
                                          kernel_size=kernel_size,
                                          in_channels=embedding_dim,
                                          out_channels=filter_num,
                                          padding=kernel_size//2)
                self._2maxpooling = nn.MaxPool1d(kernel_size=kernel_size, stride= kernel_size)

                self._3conv1d = nn.Conv1d(stride=1,
                                          kernel_size=kernel_size,
                                          in_channels=256,
                                          out_channels=filter_num,
                                          padding=kernel_size//2)

                self._4maxpooling = nn.MaxPool1d(kernel_size=kernel_size, stride= kernel_size)

                self._5conv1d = nn.Conv1d(stride=1,
                                          kernel_size=kernel_size,
                                          in_channels=256,
                                          out_channels=filter_num,
                                          padding=kernel_size//2)

                self._6maxpooling = nn.MaxPool1d(kernel_size=kernel_size*2, stride= 4)

                self._7flattern = nn.Flatten()

            def forward(self, x):
                
                batch_size, seq_len, embedding_dim  = x.shape
                x = x.permute(0,2,1)

                x = self._1conv1d(x)
                x = self._2maxpooling(x)
                x = self._3conv1d(x)
                x = self._4maxpooling(x)
                #print(x.shape)
                x = self._5conv1d(x)
                x = self._6maxpooling(x)
                x = self._7flattern(x)
                #print('cnn x shape',x.shape)
                return x

class AppNetModel(nn.Module):
    def __init__(self, payload_sz, payload_embed_sz, packet_nb, packet_embed_sz, class_nb,  lstm_layer_nb=2):
        super(AppNetModel, self).__init__()
        self.payload_sz = payload_sz
        self.payload_embed_sz = payload_embed_sz
        self.packet_nb = packet_nb
        self.packet_embed_sz = packet_embed_sz

        self.payload_embed_layer = nn.Embedding(num_embeddings=512, embedding_dim=payload_embed_sz)
        self.packet_embed_layer = nn.Embedding(num_embeddings=3200, embedding_dim=packet_embed_sz)

        self.lstm_encoder = nn.LSTM(input_size = packet_embed_sz,
                                    hidden_size= packet_embed_sz,
                                    num_layers=lstm_layer_nb,
                                    bidirectional=True, batch_first=True)

        self.cnn_encoder = CNN_block(payload_embed_sz)

        self.fc = nn.Linear(in_features=5632, out_features= class_nb)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, packet_size, payload):
        batch_size= packet_size.shape[0]
        try:
           packet_embed = self.packet_embed_layer(packet_size)
        except BaseException as exp:
           print(packet_size)
           print(exp)
        #print('packet embed shape', packet_embed.shape)
        packet_vector, hidden = self.lstm_encoder(packet_embed)
        #print('packet vector shape',packet_vector.shape)
        packet_vector = packet_vector.reshape(batch_size, -1)

        payload_embed = self.payload_embed_layer(payload)
        payload_vector = self.cnn_encoder(payload_embed)

        representation = th.cat((packet_vector, payload_vector), dim=1)
        representation = self.dropout(self.fc(representation))
        return representation