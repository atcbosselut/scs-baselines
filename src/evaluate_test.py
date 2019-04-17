import sys
import os
import random
import time
import socket
import itertools
sys.path.append(os.getcwd())

from jobman import DD

import torch

import src.training.train as train
import src.models.baselines as baselines
import src.data.data as data
import utils.utils as utils
import src.data.config as cfg

# python src/evaluate_class.py $1 $2
# $1 = {plutchik, maslow, reiss}
# $2 = gpu_index if to run on GPU

# model to run must be saved in config/predict_list_$1.txt

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

splits = ["test"]
split = splits[0]

config_file = "config/class_config.json"
config = cfg.read_config(cfg.load_config(config_file))

print("Loading model from: {}".format(config.load_model_name))
loaded_model = data.load_checkpoint(config.load_model_name)
opt = loaded_model["opt"]

print("Doing task: {}".format(opt.task))
print("Doig granularity: {}".format(opt.granularity))

if opt.net.enc.model in ["ren", "npn"]:
    data_loader = data.MemoryModelDataLoader()
    data_loader.load_vocabs(vocab_paths, vocab_text)
    data_loader = data_loader.load_data(
        opt, splits, opt.task, dl_type="memory",
        granularity=opt.granularity, exist=True)
else:
    data_loader = data.NeuralModelDataLoader()
    data_loader.load_vocabs(vocab_paths, vocab_text)
    data_loader = data_loader.load_data(
        opt, splits, opt.task,
        granularity=opt.granularity, exist=True)

data_loader.reset_offsets()

torch.manual_seed(opt.train.static.seed)
random.seed(opt.train.static.seed)
if config.gpu_mode:
    torch.cuda.manual_seed_all(opt.train.static.seed)

# Set class weight labels
data_loader.class_weights = {}
cw = 0
num = 0

# Do weighted cross-entropy based on label weights
for key in data_loader.labels[split]:
    cw += data_loader.labels[split][key].sum(0)
    num += data_loader.labels[split][key].size(0)
cw /= num
cw = 1 - torch.exp(-torch.sqrt(cw))

data_loader.class_weights[split] = cw

# Build Model
print("Building Model")
model = baselines.ClassModel(
    opt.net, data_loader.vocabs["sentence"]._tokens,
    ent_vocab=data_loader.vocabs["entity"]._tokens,
    max_words=data_loader.max_words)

print("Loading Model Parameters")
model.load_state_dict(loaded_model["state_dict"])
print("Done.")

if config.gpu_mode:
    print("Pushing to GPU")
    cfg.device = config.gpu_index
    cfg.do_gpu = True
    torch.cuda.set_device(config.gpu_index)
    data_loader.cuda(cfg.device)
    for split in data_loader.class_weights:
        data_loader.class_weights[split] = \
            data_loader.class_weights[split].cuda(
                cfg.device)

    model.cuda(cfg.device)


print("Done.")

print("Training")
_, scores, counts = train.do_class_eval_run(
    data_loader, opt, model, config.gpu_mode, {},
    opt.train.dynamic.epoch, split)
