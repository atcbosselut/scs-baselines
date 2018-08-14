import sys
import os
import random
import time

sys.path.append(os.getcwd())

import torch

import src.training.train as train
import src.models.baselines as baselines
import src.data.gen_data as data
import utils.utils as utils
import src.data.config as cfg

config_file = "config/gen_config.json"
config = cfg.read_config(cfg.load_config(config_file))
opt, meta = cfg.get_parameters(config, "gen")

vocab_paths = {
    "plutchik": "data/vocabs/plutchik8.vocab",
    "plutchik16": "data/vocabs/plutchik16.vocab",
    "reiss": "data/vocabs/reiss.vocab",
    "maslow": "data/vocabs/maslow.vocab",
}

vocab_paths["entity"] = "data/vocabs/entities_with_train.vocab"
vocab_paths["emotion"] = "data/vocabs/emotion.vocab"
vocab_paths["motivation"] = "data/vocabs/motivation.vocab"
vocab_paths["tokens"] = "data/vocabs/tokens.vocab"

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

print("Loading Data")
print opt

# Don't save models
meta.save = True
meta.do_gen = True

torch.manual_seed(opt.train.static.seed)
random.seed(opt.train.static.seed)
if config.gpu_mode:
    torch.cuda.manual_seed_all(opt.train.static.seed)

opt.train.dynamic.epoch = 0

# Build Model
print("Building Model")
print "Doing task: {}".format(opt.task)

# Initialize data loader
if opt.net.enc.model in ["ren", "npn"]:
    data_loader = data.MemoryGenModelDataLoader()
    data_loader.load_vocabs(vocab_paths, vocab_text)
    data_loader = data_loader.load_data(
        opt, splits, opt.task, dl_type="memory",
        exist=True, pruned=True)
else:
    data_loader = data.NeuralGenModelDataLoader()
    data_loader.load_vocabs(vocab_paths, vocab_text)
    data_loader = data_loader.load_data(
        opt, splits, opt.task, exist=True,
        pruned=True)

# Reset offsets
data_loader.reset_offsets(split="train", pn="pos")
data_loader.reset_offsets(split="train", pn="neg")
data_loader.reset_offsets(split="dev", pn="pos")
data_loader.reset_offsets(split="dev", pn="neg")

# Update configuration
opt.net.gendec.vSize = data_loader.vocabs[opt.task].size

if opt.net.enc.model in ["ren", "npn"] and opt.net.enc.tied:
    opt.net.enc.ents = data_loader.vocabs["entity"].size

opt.net.enc.vSize = data_loader.vocabs["sentence"].size

# Initialize Model
model = baselines.GenModel(
    opt.net, data_loader.vocabs["sentence"]._tokens,
    ent_vocab=data_loader.vocabs["entity"]._tokens,
    max_words=data_loader.max_words)

print("Done.")

print("Files will be logged at: {}".format(
    utils.make_name(opt, prefix="results/gen_losses/", is_dir=True)))

start = time.time()

# Push to GPU
if config.gpu_mode:
    print("Pushing to GPU")
    cfg.device = config.gpu_index
    cfg.do_gpu = True
    torch.cuda.set_device(config.gpu_index)
    data_loader.cuda(cfg.device)
    print "Converting data to gpu took {} s".format(time.time() - start)
    model.cuda(cfg.device)
    print "Converting model to gpu took {} s".format(time.time() - start)

print("Done.")

print("Training")
train.train_function[opt.exp](
    opt, meta, model, data_loader, config.gpu_mode)
