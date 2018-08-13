import torch, torch.nn as nn
from torch.autograd import Variable
from constants import *
import numpy as np
import math


def get_variable(x):
    if CUDA:
        return Variable(x.cuda(), requires_grad=False)
    else:
        return Variable(x, requires_grad=False)


def beat_f(x):
    """
    Returns a position encoding along the time axis
    """
    beats = torch.LongTensor(np.array([b % NOTES_PER_BAR for b in range(x.shape[1])]))
    beats = beats.repeat(((x.shape[0], x.shape[2])+(1,)))
    beats = beats.permute(0, 2, 1).contiguous()
    return get_variable(beats)


def pitch_pos_in_f(x):
    """
    Returns a constant containing pitch position of each note
    """
    pos_in = torch.LongTensor(np.arange(NUM_NOTES))
    pos_in = pos_in.repeat(x.shape[:-2]+(1,))
    return get_variable(pos_in)


def pitch_class_in_f(x):
    """
    Returns a constant containing pitch class of each note
    """
    pitch_class_matrix = np.array([n % OCTAVE for n in range(NUM_NOTES)])
    pitch_class_matrix = torch.LongTensor(pitch_class_matrix)
    pitch_class_matrix = pitch_class_matrix.repeat((x.shape[:2]+(1,)))
    return get_variable(pitch_class_matrix)


def pitch_bins_f(x):
    """
    Returns a constant containing information about pitch class usage
    """
    bins = [x[:, :, i::OCTAVE, :1].sum(2) for i in range(OCTAVE)]
    bins = torch.cat(bins, dim=-1)
    bins = bins.repeat(NUM_OCTAVES, 1, 1)
    bins = bins.view(x.shape[:2]+(NUM_NOTES, 1))
    return bins
    
    
class feature_generation(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()        
        self.padding = nn.ZeroPad2d(((2 * OCTAVE - 1)//2,  math.ceil((2 * OCTAVE - 1)/2), 0, 0))
        self.conv = nn.Conv1d(NOTE_UNITS,  OCTAVE_UNITS, 2 * OCTAVE)
        
        self.pos_embedding =nn.Embedding(num_embeddings=NUM_NOTES, embedding_dim=20)
        self.pitch_class_embedding =nn.Embedding(num_embeddings=OCTAVE, embedding_dim=OCTAVE)
        
    def forward(self, notes):
        """
        args:
            notes - tensor (batch, time, pitch, features)
        returns:
            note_features - a tensor (batch, time, pitch, embedding) that represent embedding of the piece of music
        """
        initial_shape = notes.shape
        
        # convolution
        notes = notes.contiguous().view((-1,)+notes.shape[-2:]).contiguous()
        notes = notes.permute(0, 2, 1).contiguous()
        notes = self.padding(notes)
        notes = self.conv(notes)
        notes = nn.Tanh()(notes)
        notes = notes.permute(0, 2, 1).contiguous()
        notes = notes.contiguous().view(initial_shape[:2] + notes.shape[-2:])
        
        pos_in = pitch_pos_in_f(notes)
        class_in = pitch_class_in_f(notes)
        bins = pitch_bins_f(notes)

        pos_in = self.pos_embedding(pos_in)
        class_in = self.pitch_class_embedding(class_in)
        
        note_features = torch.cat([notes, pos_in, class_in, bins], dim=-1)
    
        return note_features

