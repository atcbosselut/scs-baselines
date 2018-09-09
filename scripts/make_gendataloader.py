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

for experiment in ["emotion", "motivation"]:
    print("Making {} data for {} class of models".format(
        experiment, sys.argv[1]))

    opt = DD()
    opt.data = DD()
    opt.data.pruned = True

    # Make a memory model (EntNet, NPN) data loader for generation
    if sys.argv[1] == "memory":
        # Make save name
        name = "processed/{}/{}_{}_data_loader.pth".format(
            experiment, "gen_memory", "-".join(splits))
        print (name)

        # Initialize data loader and load vocabs and raw data
        data_loader = data.MemoryGenModelDataLoader()
        data_loader.load_vocabs(vocab_paths, vocab_text)
        data_loader.load_data(
            opt, splits, type_=experiment, pruned=opt.data.pruned)

        # Compute maximum number of words in input sentences
        # (useful for later model initialization)
        max_words = 0
        for split_data in data_loader.ctx_maxes.values():
            max_words = max(max_words, max(split_data.values()))
    # Make a neural (rnn, cnn) data loader for generation
    else:
        # Make save name
        name = "processed/{}/{}_{}_data_loader.pth".format(
            experiment, "gen_neural", "-".join(splits))
        print(name)

        # Initialize data loader and load vocabs and raw data
        data_loader = data.NeuralGenModelDataLoader()
        data_loader.load_vocabs(vocab_paths, vocab_text)
        data_loader.load_data(
            opt, splits, type_=experiment, pruned=opt.data.pruned)

        # Compute maximum number of words in input sentences
        # (useful for later model initialization)
        max_words = max(data_loader.ctx_maxes.values())

    max_words = max(max(data_loader.sent_maxes.values()), max_words)
    data_loader.max_words = max_words

    torch.save(data_loader, open(name, "w"))
