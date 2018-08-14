import os
import sys

sys.path.append(os.getcwd())

import src.data.gen_data as data

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
    "entity": "data/vocabs/entities_with_train.vocab"
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

splits = ["train", "dev", "test"]

for word in ["emotion", "motivation"]:
    print("Making {} data for {} class of models".format(
        word, sys.argv[1]))

    opt = DD()
    opt.data = DD()
    opt.data.pruned = True

    if sys.argv[1] == "memory":
        name = "processed/{}/{}_{}_data_loader.pth".format(
            word, "gen_memory", "-".join(splits))
        print (name)
        x = data.MemoryGenModelDataLoader()
        x.load_vocabs(vocab_paths, vocab_text)
        x.load_data(opt, splits, type_=word, pruned=opt.data.pruned)
        max_words = 0
        for split_data in x.ctx_maxes.values():
            max_words = max(max_words, max(split_data.values()))
    else:
        name = "processed/{}/{}_{}_data_loader.pth".format(
            word, "gen_neural", "-".join(splits))
        print(name)
        x = data.NeuralGenModelDataLoader()

        x.load_vocabs(vocab_paths, vocab_text)
        x.load_data(opt, splits, type_=word, pruned=opt.data.pruned)
        max_words = max(x.ctx_maxes.values())
    max_words = max(max(x.sent_maxes.values()), max_words)
    x.max_words = max_words
    torch.save(x, open(name, "w"))
