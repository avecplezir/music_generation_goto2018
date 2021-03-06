import torch,torch.nn as nn
import torch.nn.functional as F
from constants import *
from model.generator import *
from model.modules.feature_generation_module import feature_generation


class LstmDiscriminator(nn.Module):
    def __init__(self, hidden_size=300, last_dim=3):
        super(self.__class__, self).__init__()
        self.last_dim = last_dim
        self.hidden_size = hidden_size
        self.note_lstm = nn.LSTM(input_size=NUM_OCTAVES*last_dim, hidden_size=hidden_size)
        self.time_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self,data):
        batch_size, _, _, _ = data.size()
        octave_data = data.view(batch_size, SEQ_LEN, OCTAVE, NUM_OCTAVES,self.last_dim)\
                          .view(batch_size, SEQ_LEN, OCTAVE, NUM_OCTAVES*self.last_dim)

        note_lstm_input = octave_data.view(batch_size*SEQ_LEN, OCTAVE, NUM_OCTAVES*self.last_dim).transpose(0, 1)
        note_lstm_output, _ = self.note_lstm(note_lstm_input)
        time_lstm_input = note_lstm_output[-1].view(batch_size, SEQ_LEN, self.hidden_size).transpose(0, 1)

        time_lstm_output, _ = self.time_lstm(time_lstm_input)
        dense_input = time_lstm_output[-1]
        dense_output = self.dense(dense_input)
        probs = F.sigmoid(dense_output)
        return probs

    
class FeaturedLstmDiscriminator(nn.Module):
    def __init__(self, hidden_size=1000, last_dim=2*D_MODEL):
        super(self.__class__, self).__init__()
        self.last_dim = last_dim
        self.hidden_size = hidden_size
        self.note_lstm = nn.LSTM(input_size=NUM_OCTAVES*last_dim, hidden_size=hidden_size)
        self.time_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.dense = nn.Linear(hidden_size, 1)
        self.feature_generator = feature_generation()

    def forward(self, *input_data):
        features = []
        for sub_data in input_data:
            feature = self.feature_generator(sub_data)
            features.append(feature)
        data = torch.cat(features, dim=-1)

        *_, feature_num = data.size()
        assert feature_num == self.last_dim
        batch_size, _, _, _ = data.size()
        octave_data = data.view(batch_size, SEQ_LEN, OCTAVE, NUM_OCTAVES, self.last_dim)\
                          .view(batch_size, SEQ_LEN, OCTAVE, NUM_OCTAVES*self.last_dim)

        note_lstm_input = octave_data.view(batch_size*SEQ_LEN,OCTAVE,NUM_OCTAVES*self.last_dim).transpose(0, 1)
        note_lstm_output, _ = self.note_lstm(note_lstm_input)
        time_lstm_input = note_lstm_output[-1].view(batch_size,SEQ_LEN,self.hidden_size).transpose(0, 1)

        time_lstm_output, _ = self.time_lstm(time_lstm_input)
        dense_input = time_lstm_output[-1]
        dense_output = self.dense(dense_input)
        probs = F.sigmoid(dense_output)
        return probs
        
        
class LstmBaseline(nn.Module):
    def __init__(self, hidden_size=1000):
        super(self.__class__, self).__init__()
        self.hidden_size = hidden_size
        self.note_lstm = nn.LSTM(input_size=NUM_OCTAVES*3, hidden_size=hidden_size)
        self.time_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, data, _):
        batch_size, _, _, _ = data.size()
        octave_data = data.view(batch_size, SEQ_LEN, OCTAVE, NUM_OCTAVES,3)\
                          .view(batch_size, SEQ_LEN, OCTAVE, NUM_OCTAVES*3)

        note_lstm_input = octave_data.view(batch_size*SEQ_LEN, OCTAVE, NUM_OCTAVES*3).transpose(0, 1)
        note_lstm_output, _ = self.note_lstm(note_lstm_input)
        time_lstm_input = note_lstm_output[-1].view(batch_size, SEQ_LEN, self.hidden_size).transpose(0, 1)

        time_lstm_output, _ = self.time_lstm(time_lstm_input)
        dense_input = time_lstm_output[-1]
        dense_output = self.dense(dense_input)
        probs = F.sigmoid(dense_output)
        return probs
