import os

folders = ["train_data", "test_data"]

total_words = 0
total_lines = 0
total_paragraphs = 0

dirs = {"word": "UIT_HWDB_word_syn", "line": "UIT_HWDB_line_syn"}

def count(parent, dir):
    total_files = 0
    directory = os.path.join(parent, dir)
    for subdir in os.listdir(directory):
        _dir = os.path.join(directory, subdir)
        total_files += len(os.listdir(_dir)) - 1

    return total_files

for folder in folders:
    total_words += count(dirs["word"], folder)
    total_lines += count(dirs["line"], folder)
    # total_paragraphs += count(dirs["paragraph"], folder)

print(total_words)
print(total_lines)