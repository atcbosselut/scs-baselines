import sys
import os
import random
import time

sys.path.append(os.getcwd())

import torch

import src.training.train as train
import src.models.baselines as baselines
import src.data.data as data
import utils.utils as utils
import src.data.config as cfg

config_file = "config/class_config.json"
config = cfg.read_config(cfg.load_config(config_file))
opt, meta = cfg.get_parameters(config, "class")

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

print("Loading Data")
print opt

# Flag for whether or not to save models
meta.save = True

granularities = {
    "emotion": ["plutchik"],
    "motivation": ["maslow", "reiss"]
}

opt.epochs = 100

print "Doing task: {}".format(opt.task)
print "Doig granularity: {}".format(opt.granularity)

# Initialize data loader
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

# Update configurations
opt.net.classdec.vSize = \
    data_loader.vocabs[opt.granularity].size
if opt.net.enc.model in ["ren", "npn"] and opt.net.enc.tied:
    opt.net.enc.ents = data_loader.vocabs["entity"].size
opt.net.enc.vSize = data_loader.vocabs["sentence"].size

# Reset offsets
data_loader.reset_offsets()

# Set random seeds
torch.manual_seed(opt.train.static.seed)
random.seed(opt.train.static.seed)
if config.gpu_mode:
    torch.cuda.manual_seed_all(opt.train.static.seed)

opt.train.dynamic.epoch = 0

# Split dev set in to training and development
data_loader.split_set(
    opt.train.static.tr, keep="dev", write="train")
data_loader.reset_offsets("train")

# Set class weight labels
data_loader.class_weights = {}
for split in data_loader.labels:
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

print("Done.")

print("Files will be logged at: {}".format(
    utils.make_name(opt, prefix="results/losses/", is_dir=True)))

start = time.time()

if config.gpu_mode:
    print("Pushing to GPU")
    cfg.device = config.gpu_index
    cfg.do_gpu = True
    torch.cuda.set_device(config.gpu_index)
    data_loader.cuda(cfg.device)
    for split in data_loader.class_weights:
        data_loader.class_weights[split] = \
            data_loader.class_weights[split].cuda(cfg.device)
    print "Data to gpu took {} s".format(time.time() - start)
    model.cuda(cfg.device)
    print "Model to gpu took {} s".format(time.time() - start)

print("Done.")

print("Training")
train.train_function[opt.exp](
    opt, meta, model, data_loader, config.gpu_mode)
