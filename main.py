import os
import argparse

import torch
import torch.nn as nn
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import *
from models import RNN


DATA_FILES = os.path.join('data', 'languages', '*.txt')
N_ITERS = 1000000


def train(model, criterion, category_tensor, line_tensor, lr=0.005):
    hidden = model.init_hidden()
    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in model.parameters():
        p.data.add_(-lr, p.grad.data)

    return output, loss.item()


def evaluate(line_tensor, model):
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model.forward(line_tensor[i], hidden)

    return output


if __name__ == "__main__":
    category_lines, all_categories = assemble_data(DATA_FILES)
    n_categories = len(all_categories)
    n_hidden = 128
    n_letters = len(string.ascii_letters + ",.;")

    rnn = RNN(n_letters, n_hidden, n_categories)
    criterion = nn.NLLLoss()

    current_loss = 0.
    all_losses = []
    desc = 'Loss: %.4f | Word: %s  Predicted Language: %s %s'
    pbar = trange(N_ITERS, desc=desc, unit=' names', initial=1)
    for i in pbar:
        category, line, category_tensor, line_tensor = random_training_example(category_lines)
        output, loss = train(rnn, criterion, category_tensor, line_tensor)
        current_loss += loss
        guess_i = category_from_output(output)
        guess = all_categories[guess_i]
        correct = '✓' if guess == category else '✗ (%s)' % category
        if i % 1000 == 0:
            pbar.set_description(desc % (loss, line, guess, correct))
            all_losses.append(current_loss/1000)
            current_loss = 0.

    torch.save(rnn, 'checkpoints/language-classifier.pth')

    plt.figure()
    plt.plot(all_losses)

    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    for j in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_example(category_lines)
        output = evaluate(line_tensor, rnn)
        guess_i = category_from_output(output)
        guess = all_categories[guess_i]
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    for k in range(n_categories):
        confusion[k] = confusion[k] / confusion[k].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


