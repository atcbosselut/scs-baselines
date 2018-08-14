import os
import sys

sys.path.append(os.getcwd())

import src.data.data as data

import torch
from jobman import DD

vocab_paths = {
    "emotion": "data/vocabs/emotion.vocab",
    "motivation": "data/vocabs/motivation.vocab",
    "sentence": "data/vocabs/tokens.vocab",
    "plutchik": "data/vocabs/plutchik8.vocab",
    "plutchik16": "data/vocabs/plutchik16.vocab",
    "reiss": "data/vocabs/reiss.vocab",
    "maslow": "data/vocabs/maslow.vocab",
    "entity": "data/vocabs/entities.vocab"
}

vocab_text = {
    "emotion": (True, True),
    "motivation": (True, True),
    "sentence": (True, True),
    "reiss": (False, False),
    "maslow": (False, False),
    "plutchik": (False, False),
    "plutchik16": (False, False),
    "entity": (True, False)
}

splits = ["dev", "test"]

for i in range(3):
    if i == 0:
        word = "emotion"
        classify = "plutchik"
    elif i == 1:
        word = "motivation"
        classify = "maslow"
    elif i == 2:
        word = "motivation"
        classify = "reiss"

    print("Making {} labels for {} data for {} class of models".format(
        word, classify, sys.argv[1]))

    opt = DD()
    opt.data = DD()
    opt.data.label = "majority"

    if sys.argv[1] == "memory":
        x = data.MemoryModelDataLoader()
        x.load_vocabs(vocab_paths, vocab_text)
        x.load_data(opt, splits, type_=word, granularity=classify)
        name = "processed/{}/{}_{}_{}_{}_data_loader.pth".format(
            word, opt.data.label, "memory", classify, "-".join(splits))
        max_words = 0
        for split_data in x.ctx_maxes.values():
            max_words = max(max_words, max(split_data.values()))
    else:
        x = data.NeuralModelDataLoader()
        x.load_vocabs(vocab_paths, vocab_text)
        x.load_data(opt, splits, type_=word, granularity=classify)
        name = "processed/{}/{}_{}_{}_{}_data_loader.pth".format(
            word, opt.data.label, "neural", classify, "-".join(splits))
        max_words = max(x.ctx_maxes.values())
    max_words = max(max(x.sent_maxes.values()), max_words)
    x.max_words = max_words
    torch.save(x, open(name, "w"))
