import torch
import torch.nn as nn
from model.lstm import LSTM

'''
For variable length lstm use padding with unknown '<unk>' token
'''
class VariableLengthLSTM(nn.Module):

    def __init__(self, config):

        super(VariableLengthLSTM, self).__init__()

        self.num_hidden = int(config["no_hidden_LSTM"]/2)
        self.depth = int(config["no_LSTM_cell"])
        self.word_emb_len = 2*int(config["word_embedding_dim"])
        self.cln = config["cln"]["use_cln"]

        self.lstm = LSTM(input_size=self.word_emb_len, 
                         hidden_size=self.num_hidden, 
                         num_layers=self.depth, 
                         batch_first=True, 
                         dropout=0, 
                         cln=self.cln)

    def forward(self, word_emb, image_emb=None):
        batch_size = word_emb.size(0)
        out = self.lstm(word_emb, self.lstm.init_hidden(batch_size), image_emb)
        return out