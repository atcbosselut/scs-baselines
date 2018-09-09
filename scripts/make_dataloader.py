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
        experiment = "emotion"
        label_set = "plutchik"
    elif i == 1:
        experiment = "motivation"
        label_set = "maslow"
    elif i == 2:
        experiment = "motivation"
        label_set = "reiss"

    print("Making {} labels for {} data for {} class of models".format(
        experiment, label_set, sys.argv[1]))

    opt = DD()
    opt.data = DD()
    opt.data.label = "majority"

    # Make a memory model (EntNet, NPN) data loader for generation
    if sys.argv[1] == "memory":
        # Initialize data loader and load vocabularies and data
        data_loader = data.MemoryModelDataLoader()
        data_loader.load_vocabs(vocab_paths, vocab_text)
        data_loader.load_data(
            opt, splits, type_=experiment, granularity=label_set)

        # Make save name
        name = "processed/{}/{}_{}_{}_{}_data_loader.pth".format(
            experiment, opt.data.label, "memory", label_set, "-".join(splits))

        # Compute maximum number of words in input sentences
        # (useful for later model initialization)
        max_words = 0
        for split_data in data_loader.ctx_maxes.values():
            max_words = max(max_words, max(split_data.values()))
    # Make a neural (rnn, cnn) data loader for generation
    else:
        # Initialize data loader and load vocabularies and data
        data_loader = data.NeuralModelDataLoader()
        data_loader.load_vocabs(vocab_paths, vocab_text)
        data_loader.load_data(
            opt, splits, type_=experiment, granularity=label_set)

        # Make save name
        name = "processed/{}/{}_{}_{}_{}_data_loader.pth".format(
            experiment, opt.data.label, "neural", label_set, "-".join(splits))

        # Compute maximum number of words in input sentences
        # (useful for later model initialization)
        max_words = max(data_loader.ctx_maxes.values())

    max_words = max(max(data_loader.sent_maxes.values()), max_words)
    data_loader.max_words = max_words

    # Save data loader for quicker loading in experiments
    torch.save(data_loader, open(name, "w"))
