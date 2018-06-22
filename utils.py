from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import os
import random

import torch


def find_files(path):
    return glob.glob(path)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in string.ascii_letters
    )


def read_lines(filename):
    with open(filename, encoding='utf-8') as file:
        lines = file.read().strip().split('\n')

    return lines


def assemble_data(data_glob):
    category_lines = {}
    all_categories = []
    for file in find_files(data_glob):
        category = os.path.basename(file).split('.')[0]
        all_categories.append(category)
        lines = read_lines(file)
        category_lines[category] = lines

    return category_lines, all_categories


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()

    return category_i

def letter_to_index(letter):
    all_letters = string.ascii_letters + '.,;'

    return all_letters.find(letter)


def letter_to_tensor(letter):
    n_letters = len(string.ascii_letters + '.,;')
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1

    return tensor


def line_to_tensor(line):
    n_letters = len(string.ascii_letters + '.,;')
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1

    return tensor


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example(data):
    all_categories = list(data.keys())
    category = random_choice(all_categories)
    line = random_choice(data[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)

    return category, line, category_tensor, line_tensor


def evaluate(line_tensor, model):
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model.forward(line_tensor[i], hidden)

    return output
