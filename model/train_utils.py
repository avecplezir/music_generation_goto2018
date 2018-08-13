import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import time
from IPython import display
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim
from constants import *
from utils.music_metrics import EB, UPC, QN

criterion_bce_play = nn.BCELoss()
criterion_bce_replay = nn.BCELoss()


def compute_loss(y_pred, y_true, time_to_punish=8):
    """
    last dimetion of y_pred and y_true represent play, replay and volume
    args:
        y_pred - tensor (batch, time, pitch, 3)
        y_true - tensor (batch, time, pitch, 3)
        time_to_punish - integer (mask loss before the time_to_punish timestep)
    returns:
        loss: float
    """

    y_pred = y_pred[:, time_to_punish:, :, :]
    y_true = y_true[:, time_to_punish:, :, :]

    played = y_true[:, :, :, 0]

    bce_note = criterion_bce_play(y_pred[:, :, :, 0], y_true[:, :, :, 0])
    replay = played * y_pred[:, :, :, 1] + (1 - played) * y_true[:, :, :, 1]
    bce_replay = criterion_bce_replay(replay, y_true[:, :, :, 1])

    return bce_note + bce_replay


def iterate_minibatches(train_data, train_labels, batchsize):
    indices = np.random.permutation(np.arange(len(train_labels)))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]

        if CUDA:
            yield Variable(torch.FloatTensor(train_data[ix])).cuda(), Variable(
                torch.FloatTensor(train_labels[ix])).cuda()
        else:
            yield Variable(torch.FloatTensor(train_data[ix])), Variable(torch.FloatTensor(train_labels[ix]))


def plot_history(history):
    plt.subplot(221)
    plt.title("losses")
    plt.xlabel("#epoch")
    plt.ylabel("loss")
    plt.plot(history['train_loss'], 'b', label='train_loss')
    plt.plot(history['val_loss'], 'g', label='val_loss')
    plt.legend()
    plt.subplot(222)
    plt.plot(history['EB_true'], label="EB_true")
    plt.plot(history['EB_false'], label="EB_false")
    plt.legend()
    plt.subplot(223)
    plt.plot(history['UPC_true'], label="UPC_true")
    plt.plot(history['UPC_false'], label="UPC_false")
    plt.legend()
    plt.subplot(224)
    plt.plot(history['QN_true'], label="QN_true")
    plt.plot(history['QN_false'], label="QN_false")
    plt.legend()
    plt.show()


def train(generator, X_tr, X_te, y_tr, y_te, batchsize=3, n_epochs=3, verbose=True):
    """
    args:
        generator: model
        X_tr, X_te, y_tr, y_te: datasets
        batchsize: int
    returns:
        generator: model
        epoch: integer (the last epoch)
        epoch_history: dict
    """
    generator.note_ax.to_train = True
    generator.note_ax.apply_T = False

    optimizer = optim.Adam(generator.parameters())
    n_train_batches = math.ceil(len(X_tr) / batchsize)
    n_validation_batches = math.ceil(len(X_te) / batchsize)

    history = {'train_loss': [], 'val_loss': [],
               'EB_true': [], 'UPC_true': [], 'QN_true': [], 'EB_false': [], 'UPC_false': [], 'QN_false': []}

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = 0
        generator.train(True)

        try:
            for X, y in tqdm(iterate_minibatches(X_tr, y_tr, batchsize)):
                pred, sound = generator(X, y)
                loss = compute_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data.numpy()

            train_loss /= n_train_batches

            generator.train(False)
            val_loss = 0
            for X, y in tqdm(iterate_minibatches(X_te, y_te, batchsize)):
                pred, sound = generator(X, y)
                loss = compute_loss(pred, y)

                val_loss += loss.cpu().data.numpy()

            val_loss /= n_validation_batches

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            # metrics computed on the last val batch
            history['EB_true'].append(EB(y).mean().data.cpu().numpy())
            history['UPC_true'].append(UPC(y).mean().data.cpu().numpy())
            history['QN_true'].append(QN(y).mean().data.cpu().numpy())
            history['EB_false'].append(EB(sound).mean().data.cpu().numpy())
            history['UPC_false'].append(UPC(sound).mean().data.cpu().numpy())
            history['QN_false'].append(QN(sound).mean().data.cpu().numpy())

        except KeyboardInterrupt:
            return generator, epoch, history

            # Visualize
        if verbose:
            display.clear_output(wait=True)
            plt.figure(figsize=(16, 6))
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print('current train loss: {}'.format(history['train_loss'][-1]))
            print('current val loss: {}'.format(history['val_loss'][-1]))
            plot_history(history)

    print("Finished!")

    return generator, epoch, history