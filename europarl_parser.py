import os
from utils import read_lines, unicode_to_ascii

from tqdm import tqdm


def parse_file(file):

    lines = read_lines(file)
    out = []
    pbar = tqdm(lines, unit=" line", desc="Reading {}".format(file))
    for line in pbar:
        words = [unicode_to_ascii(s) for s in line.split()]
        for word in words:
            if word:
                out.append(word)

    out = set(out)
    out_file = os.path.join('data', 'languages', os.path.basename(file))
    with open(out_file, 'w') as f:
        pbar2 = tqdm(out, unit=" word", desc="Writing {}".format(out_file))
        for word in pbar2:
            f.write(word + "\n")


if __name__ == "__main__":
    raw_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'languages')
    for file in os.listdir(raw_dir):
        if file not in os.listdir(processed_dir):
            file_path = os.path.join('data', 'raw', file)
            parse_file(file_path)
