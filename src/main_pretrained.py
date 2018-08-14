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


config_file = "config/pretrained_config.json"
config = cfg.read_config(cfg.load_config(config_file))
opt, meta = cfg.get_parameters(config, "joint")

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

# Don't save models
meta.save = True

opt.epochs = 100

# Load model
print 'Loading model from: {}'.format(
    config["load_model_{}_{}".format(opt.net.enc.model, opt.task)])
loaded_model = data.load_checkpoint(
    config["load_model_{}_{}".format(opt.net.enc.model, opt.task)])

# Load configuration
old_opt = loaded_model["opt"]

if old_opt.net.enc.model != opt.net.enc.model:
    print "Not the same model being run. Ending"
    raise

# Save number of epochs trained for pretrained model
# Good for tracking the pretrained models we used
opt.net.enc["prev#"] = config["load_model_{}_{}".format(
    opt.net.enc.model, opt.task)][:-7].split("_")[-1]

opt.net.gendec = old_opt.net.gendec

# Assure that the task is for one that was pretrained
opt.task = old_opt.task

print "Doing task: {}".format(opt.task)
print "Doig granularity: {}".format(opt.granularity)

# Initialize data loaders
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

# update configuration
opt.net.classdec.vSize = \
    data_loader.vocabs[opt.granularity].size

if opt.net.enc.model in ["ren", "npn"] and opt.net.enc.tied:
    opt.net.enc.ents = data_loader.vocabs["entity"].size

opt.net.enc.vSize = data_loader.vocabs["sentence"].size

# Reset offsets in data loader
data_loader.reset_offsets()

# Initialize random seeds
torch.manual_seed(opt.train.static.seed)
random.seed(opt.train.static.seed)
if config.gpu_mode:
    torch.cuda.manual_seed_all(opt.train.static.seed)

opt.train.dynamic.epoch = 0

# Split development set into training set and remaining dev set
data_loader.split_set(opt.train.static.tr, keep="dev", write="train")
data_loader.reset_offsets("train")

data_loader.class_weights = {}

# Initialize class weights for training
for split in data_loader.labels:
    cw = 0
    num = 0
    for key in data_loader.labels[split]:
        cw += data_loader.labels[split][key].sum(0)
        num += data_loader.labels[split][key].size(0)
    cw /= num
    cw = 1 - torch.exp(-torch.sqrt(cw))

    data_loader.class_weights[split] = cw

# Hacky, but don't change this
# Explanation:
# The longest sequence to encode in the generation model is 21 tokens long.
# The longest sequence to encode in the classification task is less than that.
#
# If we use the classification data_loaders to set the maximum number of words
# to encode for the model, it won't be able to load the pretrained model
# because the parameter dimensions won't match up.
# This shouldn't be a problem, because it will never use the extra parameters
max_words = 21

# Build Model
print("Building Model")
model = baselines.ClassModel(
    opt.net, data_loader.vocabs["sentence"]._tokens,
    ent_vocab=data_loader.vocabs["entity"]._tokens,
    max_words=max_words)
temp_model = baselines.GenModel(
    old_opt.net, data_loader.vocabs["sentence"]._tokens,
    ent_vocab=data_loader.vocabs["entity"]._tokens,
    max_words=max_words)

# Save previous information if it was different
# Good for tracking the pretrained models we used
for k, v in old_opt.net.enc.iteritems():
    if opt.net.enc[k] != v:
        opt.net.enc[k + "#"] = v

# Load state dictionary into model architecture
print("Loading Model Parameters")
temp_model.load_state_dict(loaded_model["state_dict"])
print("Done.")

# Update models
model.encoder = temp_model.encoder
model.ent_encoder = temp_model.ent_encoder

print("Done.")

print("Files will be logged at: {}".format(
    utils.make_name(opt, prefix="results/losses/",
                    is_dir=True)))

start = time.time()

# Push to GPU
if config.gpu_mode:
    print("Pushing to GPU")
    cfg.device = config.gpu_index
    cfg.do_gpu = True

    torch.cuda.set_device(cfg.device)
    data_loader.cuda(cfg.device)
    for split in data_loader.class_weights:
        data_loader.class_weights[split] = \
            data_loader.class_weights[split].cuda(
                cfg.device)

    print "Data to gpu took {} s".format(
        time.time() - start)
    model.cuda(cfg.device)
    print "Model to gpu took {} s".format(
        time.time() - start)

print("Done.")

print("Training")
train.train_function[opt.exp](
    opt, meta, model, data_loader, config.gpu_mode)
