from functions import CreateLineImgDataset, CreateWordImgDataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--level", type=str, help="word level or line level")
parser.add_argument("--start-from", type=int, default=0)

args = parser.parse_args()
args = vars(args)

corpus = {"word": "VN_word_dataset.txt", "line": "VN_line_dataset.txt"}

if args["level"] == "word":
    words = open(corpus["word"]).readlines()
    # words.sort()
    CreateWordImgDataset(words, args["start_from"])
else:
    lines = open(corpus["line"]).readlines()
    # lines.sort()
    CreateLineImgDataset(lines, args["start_from"])