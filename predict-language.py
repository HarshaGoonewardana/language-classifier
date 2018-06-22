import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument('word', type=str)
parser.add_argument('--n-predictions', type=int, required=False, default=3)
args = parser.parse_args()

rnn = torch.load('checkpoints/language-classifier.pth')
DATA_FILES = 'data/languages/*.txt'


def assemble_data(data_glob, n=10000):
    category_lines = {}
    all_categories = []
    for file in find_files(data_glob):
        category = os.path.basename(file).split('.')[0]
        all_categories.append(category)
        lines = read_lines(file)
        random.shuffle(lines)
        category_lines[category] = lines[:n] if n < len(lines) else lines

    return category_lines, all_categories


def evaluate(line_tensor, model):
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model.forward(line_tensor[i], hidden)

    return output


def line_to_tensor(line):
    n_letters = len(string.ascii_letters)
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1

    return tensor


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
