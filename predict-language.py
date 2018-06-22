#!/usr/bin/env python36

import argparse

import torch

from utils import assemble_data, evaluate, line_to_tensor


parser = argparse.ArgumentParser()
parser.add_argument('word', type=str)
parser.add_argument('--n-predictions', type=int, required=False, default=3)

args = parser.parse_args()

rnn = torch.load('checkpoints/language-classifier.pth')
DATA_FILES = 'data/languages/*.txt'


def predict(line, n_predictions=3):
    _, all_categories = assemble_data(DATA_FILES)
    output = evaluate(line_to_tensor(line), rnn)

    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []
    print(line)
    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions


if __name__ == "__main__":
    predict(args.word, args.n_predictions)

