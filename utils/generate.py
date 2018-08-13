import numpy as np
from collections import deque
import midi

from constants import *
from utils.dataset import unclamp_midi, compute_beat
from tqdm import tqdm
from utils.midi_util import midi_encode

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MusicGeneration:
    """
    Represents a music generation
    """

    def __init__(self, default_temp=1):
        self.notes_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.beat_memory = deque(np.zeros(SEQ_LEN), maxlen=SEQ_LEN)

        # The next note being built
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR

        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_time_inputs(self):
        return (
            np.array(self.notes_memory),
            np.array(self.beat_memory),

        )

    def build_note_inputs(self, note_features):
        # Timesteps = 1 (No temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_note]),
        )

    def add_notes(self, notes):
        self.next_note = notes

    def end_time(self, t):
        """
        Finish generation for this time step.
        """

        self.notes_memory.append(self.next_note)
        # Consistent with dataset representation
        self.beat_memory.append(t % NOTES_PER_BAR)
        self.results.append(self.next_note)
        # Reset next note
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        return self.results[-1]


def process_inputs(ins):
    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins


def generate(models, num_bars, to_train=False):
    print('Generating music...')

    #     models.train(False)
    time_model, note_model = models.time_ax, models.note_ax
    note_model.to_train = to_train
    note_model.apply_T = not to_train

    generations = [MusicGeneration()]
    g = generations[0]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Produce note-invariant features
        ins, beat = process_inputs([g.build_time_inputs() for g in generations])
        beat = beat[0]

        if CUDA:
            ins = Variable(torch.FloatTensor(ins)).cuda()
            beat = Variable(torch.LongTensor(beat)).cuda()
        else:
            ins = Variable(torch.FloatTensor(ins))
            beat = Variable(torch.LongTensor(beat))

        # Pick only the last time step
        note_features = time_model(ins, beat)
        note_features = note_features[:, -1:, :]

        predictions, sample = note_model(note_features, None)
        sample = sample.cpu().data.numpy()[0][0]
        # add volume dimension
        g.add_notes(sample)

        # Move one time step
        yield [g.end_time(t) for g in generations]


def write_file(name, results):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    results = zip(*list(results))

    for i, result in enumerate(results):
        fpath = os.path.join(name + '.mid')
        print('Writing file', fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        mf = midi_encode(unclamp_midi(result))
        midi.write_midifile(fpath, mf)