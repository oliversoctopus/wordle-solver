import json
import os
import numpy as np
import itertools
from collections import Counter

from wordle_data import DATA_DIR, DERIVED_DIR, write_json

# relative to this file, not to the working directory
def derived(name):
    return os.path.join(DERIVED_DIR, name)

os.makedirs(DERIVED_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "wordles.json"), "r") as file:
    wordles = json.load(file)

with open(os.path.join(DATA_DIR, "nonwordles.json"), "r") as file:
    nonwordles = json.load(file)

# We use all possible words as answers, since sometimes the editor chooses answers outside the original 2022 answer list
all_words = wordles[:]
all_words.extend(nonwordles)

# Generate all combinations of results, save to dictionary, then save to file
results = ["N", "Y", "G"] # none, yellow, green
combinations = ["".join(p) for p in itertools.product(results, repeat=5)]
pat2num = {p: n for n, p in enumerate(combinations)}
num2pat = combinations[:]
write_json(derived("pat2num.json"), pat2num)
write_json(derived("num2pat.json"), num2pat)

word2ind = {w: i for i, w in enumerate(all_words)}
ind2word = all_words[:]
write_json(derived("word2ind.json"), word2ind)
write_json(derived("ind2word.json"), ind2word)

val = {'N': 0, 'Y': 1, 'G': 2}

def wordle_result(guess, answer):
    result = ['N']*5
    letter_counts = Counter(answer)
    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = 'G'
            letter_counts[answer[i]] -= 1
    for i in range(5):
        if result[i] != 'G' and letter_counts[guess[i]] > 0:
            result[i] = 'Y'
            letter_counts[guess[i]] -= 1
    result_num = 0
    for i in range(5):
        result_num *= 3
        result_num += val[result[i]]
    return result_num

N = len(all_words)
feedback_table = np.zeros((N, N), dtype=np.uint8) # feedback_table[guess][answer] = feedback number
for i in range(N):
    for j in range(N):
        feedback_table[i][j] = wordle_result(all_words[i], all_words[j])

np.save(derived("feedback_table.npy"), feedback_table)