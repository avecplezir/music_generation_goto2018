import torch, torch.nn as nn
from constants import *
import numpy as np

from model.modules.feature_generation_module import feature_generation, beat_f


def sample_sound(data_gen):
    """sample music from distribution"""
    size = data_gen.size()
    assert len(size) in [3, 4], 'dimension of input tensor for sampling must be 3 or 4 dimensional'
    rand = torch.from_numpy(np.random.random(size)).type(torch.FloatTensor)
    sample = (rand < data_gen).type(torch.FloatTensor)

    if len(size) == 4:
        vol = sample[:, :, :, :1]
    elif len(size) == 3:
        vol = sample[:, :, :1]
    sample = torch.cat([sample, vol], dim=-1)

    if CUDA:
        return sample.cuda()
    else:
        return sample


class time_axis(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.n_layers = TIME_AXIS_LAYERS
        self.hidden_size = TIME_AXIS_UNITS

        self.input_size = NOTE_EMBEDDING
        self.input_size += BEATS_FEATURES

        self.time_lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=0.1,
                                 batch_first=True, )
        self.dropout = nn.Dropout(p=0.2)
        self.generate_features = feature_generation()
        self.beat_embedding = nn.Embedding(num_embeddings=16, embedding_dim=16)

    def forward(self, notes, beats=None):

        """
        arg:
            notes - (batch, time_seq, note_seq, note_features)

        out:
            (batch, time_seq, note_seq, hidden_features)

        """

        notes = self.dropout(notes)

        note_features = self.generate_features(notes)
        notes = note_features

        if beats is None:
            beats = beat_f(notes)
        else:
            # used when music is generating
            beats = beats.repeat((notes.shape[2], 1, 1)).permute(1, 2, 0).contiguous()
        beats = self.beat_embedding(beats)

        notes = torch.cat([notes, beats], dim=-1)
        initial_shape = notes.shape

        notes = notes.permute(0, 2, 1, 3).contiguous()
        notes = notes.view((-1,) + notes.shape[-2:]).contiguous()

        lstm_out, hidden = self.time_lstm(notes)

        time_output = lstm_out.contiguous().view((initial_shape[0],) + (initial_shape[2],) + lstm_out.shape[-2:])
        time_output = time_output.permute(0, 2, 1, 3)

        return time_output


class note_axis(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.n_layers = NOTE_AXIS_LAYERS
        self.hidden_size = NOTE_AXIS_UNITS
        # number of time features plus number of previous higher note in the same time momemt
        self.input_size = TIME_AXIS_UNITS + NOTE_UNITS

        self.note_lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=0.1,
                                 batch_first=True, )

        self.dropout = nn.Dropout(p=0.2)

        self.logits = nn.Linear(self.hidden_size, NOTE_UNITS - 1)

        # mode
        self.to_train = True
        self.apply_T = False
        self.temperature = 1
        self.silent_time = 0

    def generate_music(self, notes):

        initial_shape = notes.shape
        note_input = notes.contiguous().view((-1,) + notes.shape[-2:]).contiguous()

        if CUDA:
            hidden = (torch.zeros(self.n_layers, note_input.shape[0], self.hidden_size).cuda(),
                      torch.zeros(self.n_layers, note_input.shape[0], self.hidden_size).cuda())
        else:
            hidden = (torch.zeros(self.n_layers, note_input.shape[0], self.hidden_size),
                      torch.zeros(self.n_layers, note_input.shape[0], self.hidden_size))

        notes_list = []
        sound_list = []
        if CUDA:
            sound = torch.zeros(note_input[:, 0:1, :].shape[:-1] + (NOTE_UNITS,)).cuda()
        else:
            sound = torch.zeros(note_input[:, 0:1, :].shape[:-1] + (NOTE_UNITS,))

        for i in range(NUM_OCTAVES * OCTAVE):

            inputs = torch.cat([note_input[:, i:i + 1, :], sound], dim=-1)
            note_output, hidden = self.note_lstm(inputs, hidden)

            logits = self.logits(note_output)
            if self.apply_T:
                next_notes = nn.Sigmoid()(logits / self.temperature)
            else:
                next_notes = nn.Sigmoid()(logits)

            sound = sample_sound(next_notes)
            notes_list.append(next_notes)
            sound_list.append(sound)

        out = torch.cat(notes_list, dim=1)
        sounds = torch.cat(sound_list, dim=1)

        if self.apply_T:
            if (sounds[-1, :, 0] != 0).sum() == 0:
                self.silent_time += 1
                if self.silent_time >= NOTES_PER_BAR:
                    self.temperature += 0.1
            else:
                self.silent_time = 0
                self.temperature = 1

        note_output = out.contiguous().view(initial_shape[:2] + out.shape[-2:])
        sounds = sounds.contiguous().view(initial_shape[:2] + sounds.shape[-2:])

        return note_output, sounds

    def forward(self, notes, chosen):
        """
        arg:
            notes: (batch, time_seq, note_seq, time_hidden_features)
            chosen: (batch, time_seq, note_seq, time_hidden_features)

        returns:
            (batch, time_seq, note_seq, next_notes_features)

        """

        if self.to_train:
            # Shift target one note to the left.
            chosen = self.dropout(chosen)
            shift_chosen = nn.ZeroPad2d((0, 0, 1, 0))(chosen[:, :, :-1, :])
            notes = torch.cat([notes, shift_chosen], dim=-1)

            initial_shape = notes.shape
            note_input = notes.contiguous().view((-1,) + notes.shape[-2:]).contiguous()

            out, hidden = self.note_lstm(note_input)
            note_output = out.contiguous().view(initial_shape[:2] + out.shape[-2:])
            logits = self.logits(note_output)
            next_notes = nn.Sigmoid()(logits)
            note_output, sounds = next_notes, sample_sound(next_notes)
        else:
            # used when music is generating
            note_output, sounds = self.generate_music(notes)

        return note_output, sounds


class Generator(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.time_ax = time_axis()
        self.note_ax = note_axis()

    def forward(self, notes, chosen=None, beat=None):
        """
        arg:
            notes: tensor
            chosen: tensor (predictions - moved notes tensor with one new column)
            beat: tensor
        """

        note_ax_output = self.time_ax(notes, beat)
        output = self.note_ax(note_ax_output, chosen)

        return output 
