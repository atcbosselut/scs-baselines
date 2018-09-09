"""
File for methods related to parsing data files and
initializing a data loader. Classes in this file build
specific data loaders for different models/experiments.
Each data loader has methods for shuffling its data,
sampling batches, loading examples from files, etc.

author: Antoine Bosselut (atcbosselut)
"""

import os
import ast
import cPickle as pickle
import random

import nltk
import torch
import progressbar
import pandas as pd

import src.data.config as cfg
import utils.utils as utils


# Save the model
def save_checkpoint(state, filename):
    torch.save(state, filename)


# Name the checkpoint and save the model
def save_step(model, optimizer, opt, length):
    name = "{}.pickle".format(utils.make_name(
        opt, prefix="models/", is_dir=False))
    save_checkpoint({
        'epoch': length, 'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(), "opt": opt},
        name)
    print "Saving to: {}".format(name)


# Name the training loss and save them
def save_train_losses(opt, iter_loss, ln="losses"):
    name = "{}/train.pickle".format(utils.make_name(
        opt, prefix="results/{}/".format(ln), is_dir=True))
    print "Saving training losses to: {}".format(name)
    with open(name, "w") as f:
        pickle.dump(iter_loss, f)


# Name evaluation losses and save them
def save_eval_losses(opt, eval_losses, split="dev", ln="losses"):
    if eval_losses:
        name = "{}/{}.pickle".format(utils.make_name(
            opt, prefix="results/{}/".format(ln), is_dir=True), split)
        print "Saving evaluation losses to: {}".format(name)
        with open(name, "w") as f:
            pickle.dump(eval_losses, f)


# Load a model checkpoint
def load_checkpoint(filename, gpu=True):
    if os.path.exists(filename):
        checkpoint = torch.load(
            filename, map_location=lambda storage, loc: storage)
    else:
        print("No model found at {}".format(filename))
        checkpoint = None
    return checkpoint


# Translate pandas dataframe cells
def lit_eval(s):
    if s:
        return nltk.casual_tokenize(s.lower())
    else:
        return []


# Translate pandas dataframe cells for context
def ctx_lit_eval(s):
    if isinstance(s, str):
        sents = s.split("|")
        return [nltk.casual_tokenize(i.lower()) for i in sents]
    return []


# Create a new vocabulary from the a vocabulary file
class Vocab(object):
    def __init__(self, link=None):
        self._tokens = []
        self._index = {}
        self.size = 0
        if link:
            self.load_vocab(link)

    def load_vocab(self, link, meta=True, text=False):
        if "pickle" in link:
            with open(link, "r") as f:
                temp = pickle.load(f)
        else:
            with open(link, "r") as f:
                temp = f.read().split("\n")

        # If we're including meta tokens {pad, unk}
        if meta:
            self._tokens = {i + 1: j for i, j in enumerate(temp)}
            self._tokens[0] = "<pad>"
            self._tokens[len(self._tokens)] = "<unk>"

            # If this requires start and end tokens
            if text:
                self._tokens[len(self._tokens)] = "<start>"
                self._tokens[len(self._tokens)] = "<end>"
        else:
            self._tokens = {i: j for i, j in enumerate(temp)}

        self._index = {j: i for i, j in self._tokens.iteritems()}

        self.size = len(self._tokens)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._tokens[i]
        else:
            return self._index.get(i, self._index['<unk>'])


class DataLoader(object):
    def __init__(self, opt=None, batch_size=32):
        self.is_cuda = False
        self.vocabs = {}

        self.offset = {}
        self.unfilled = {}

        self.is_ren = False

        self.batch_size = batch_size

        # File names we'll load to create the data
        self.fnames = {
            "train": ["data/training/allcharlinepairs_noids.csv",
                      "data/training/allcharlinepairs_part2_noids.csv"],
            "train_entity": "data/training/entity_lines_fast.pickle",
            "train_entity_mentions": "data/training/entity_mentions.pickle",
            "train_num_entities": "data/training/num_entities.pickle",

            "dev_entity": "data/dev/entity_lines_fast.pickle",
            "dev_entity_mentions": "data/dev/entity_mentions.pickle",
            "dev_num_entities": "data/dev/num_entities.pickle",
            "dev_emotion": "data/dev/emotion/allcharlinepairs_noids.csv",
            "dev_motivation": "data/dev/motivation/allcharlinepairs_noids.csv",

            "test_entity": "data/test/entity_lines_fast.pickle",
            "test_entity_mentions": "data/test/entity_mentions.pickle",
            "test_num_entities": "data/test/num_entities.pickle",
            "test_emotion": "data/test/emotion/allcharlinepairs_noids.csv",
            "test_motivation": "data/test/motivation/allcharlinepairs_noids.csv",

        }

    def load_vocabs(self, paths, is_texts={}):
        for vocab_name, vocab_path in paths.iteritems():
            print("{} \t {}".format(vocab_name, vocab_path))
            temp_vocab = Vocab()
            meta = is_texts.get(vocab_name, [True])[0]
            text = is_texts.get(vocab_name, [None, False])[1]
            temp_vocab.load_vocab(vocab_path, meta, text)
            self.vocabs[vocab_name] = temp_vocab

    def get_maxes(self, data, ctx=False, ents=False):
        maximum = 0
        if not ctx:
            for listed in data:
                maximum = max(len(listed), maximum)
            return maximum
        maximums = {}
        for listed in data:
            if self.is_ren:
                for i, sent in enumerate(listed):
                    maximums[i] = max(maximums.get(i, 0), len(sent))
            else:
                length = 0
                for i, sent in enumerate(listed):
                    length += len(sent)
                    maximums[i] = max(maximums.get(i, 0), length)

        return maximums

    def get_splits(self, keep, ratio):
        ids = set()
        for key in self.ids[keep]:
            ids = ids.union([i[0] for i in self.ids[keep][key]])
        idxs = list(ids)
        random.shuffle(idxs)

        diff = int(ratio * len(idxs))

        write_idx = set(idxs[:diff])
        keep_idx = set(idxs[diff:])

        keep_vals = {}
        write_vals = {}

        for key in self.ids[keep]:
            keep_vals[key] = [i for i, j in enumerate(self.ids[keep][key])
                              if j[0] in keep_idx]
            write_vals[key] = [i for i, j in enumerate(self.ids[keep][key])
                               if j[0] in write_idx]

        return keep_vals, write_vals

    def make_label_point(self, row, same_row):
        if same_row is None:
            if self.type == "emotion":
                if self.granularity == "plutchik":
                    r_labels = [e[:-2] for e in ast.literal_eval(
                        row["plutchik"]) if e != "none"]
                else:
                    r_labels = ast.literal_eval(row["plutchik"])

            else:
                r_labels = [e for e in ast.literal_eval(
                            row[self.granularity]) if e != "none" and
                            e != "na"]

            labels = [0] * self.vocabs[self.granularity].size

            for word in r_labels:
                labels[self.vocabs[self.granularity]._index[word]] = 1
        else:
            if self.type == "emotion":
                label_num = len(same_row["plutchik"])
                labels = same_row["plutchik"].apply(ast.literal_eval).sum()

                label_countss = {i: labels.count(i) for i in labels
                                 if i != "none"}

                if self.granularity == "plutchik":
                    label_counts = {}
                    for e, c in label_countss.iteritems():
                        label_counts[e[:-2]] = label_counts.get(e[:-2], 0) + 1
                else:
                    label_counts = label_countss
            else:
                label_num = len(same_row[self.granularity])
                labels = same_row[self.granularity].apply(
                    ast.literal_eval).sum()
                label_counts = {i: labels.count(i) for i in labels
                                if i not in ["none", "na"]}

            labels = [0] * self.vocabs[self.granularity].size
            r_labels = []
            for word, count in label_counts.iteritems():
                if count < (label_num / 2 + 1):
                    continue
                labels[self.vocabs[self.granularity]._index[word]] = 1
                r_labels.append(word)
        return labels, r_labels


class NeuralModelDataLoader(DataLoader):
    def __init__(self, opt=None, batch_size=32):
        super(NeuralModelDataLoader, self).__init__(opt, batch_size)

        # Emotion/Motivations Generations
        self.labels = {}

        # Context sentences
        self.ctx = {}

        # Context Lengths
        self.ctx_lengths = {}

        # Sentences
        self.sent = {}

        # Sentence Lengths
        self.sent_lengths = {}

        # Unfilled and offset trackers
        self.unfilled = {}
        self.offset = {}

        # Remember ID of each entry
        self.ids = {}

        self.raw = {}
        self.raw_labels = {}

        self.ent = {}
        self.ent_labels = {}
        self.ent_init = {}

        self.size = {}

        self.sent_maxes = {}
        self.ctx_maxes = {}

    # Load data from file
    def load_data(self, opt, splits=["train"], type_="emotion",
                  granularity="plutchik", exist=False, dl_type="neural"):
        self.splits = splits
        self.granularity = granularity
        if granularity == "plutchik" and type_ == "motivation":
            self.granularity = "maslow"
        self.type = type_

        if exist:
            name = "processed/{}/{}_{}_{}_dev-test_data_loader.pth".format(
                type_, opt.data.label, dl_type, granularity)
            print "Loading model from: {}".format(name)
            with open(name, "r") as f:
                self = torch.load(f)
            return self

        for split in splits:
            self.labels[split] = {}

            self.ctx[split] = {}
            self.ctx_lengths[split] = {}

            self.sent[split] = {}
            self.sent_lengths[split] = {}

            self.ids[split] = {}
            self.size[split] = {}
            self.raw[split] = {}
            self.raw_labels[split] = {}
            self.ent[split] = {}
            self.ent_labels[split] = {}
            self.ent_init[split] = {}

            print "Reading from Data File"
            if split == "train":
                print("Reading data from {}".format(self.fnames[split]))
                data = pd.read_csv(self.fnames[split])
            else:
                print("Reading data from {}".format(
                    self.fnames["{}_{}".format(split, type_)]))
                data = pd.read_csv(self.fnames["{}_{}".format(split, type_)])
            print("Reading entity identities from {}".format(
                self.fnames[split + "_entity"]))
            ents = pickle.load(open(
                self.fnames[split + "_entity"], "r"))
            print("Reading entity frequencies from {}".format(
                self.fnames[split + "_num_entities"]))
            num_ents = pickle.load(open(
                self.fnames[split + "_num_entities"], "r"))
            print("Reading entity mentions from {}".format(
                self.fnames[split + "_entity_mentions"]))
            ent_ments = pickle.load(open(
                self.fnames[split + "_entity_mentions"], "r"))

            print "Done reading data"
            self.sent_maxes[split] = self.get_maxes(
                data["sentence"].apply(lit_eval))
            self.ctx_maxes[split] = self.get_maxes(
                data["context"].apply(ctx_lit_eval), ctx=True)

            print "Putting Data in Tensors"
            print "Doing {}".format(split)

            bar = progressbar.ProgressBar(max_value=len(data.index))
            bar.update(0)

            # track visited story-character-line triplets
            visited = set()

            # Count of how many times a story-character-line triplet is seen
            # for the purpose of making IDs
            counter = {}

            for i in data.index:
                if opt.data.label == "majority" and i in visited:
                    continue

                row = data.loc[i]

                if opt.data.label == "majority":
                    sid = row["storyid"]
                    condition = (data["storyid"] == sid)
                    condition = condition & (data["char"] == row["char"])
                    condition = condition & (data["linenum"] == row["linenum"])
                    same_rows = data[condition]
                    self.do_row(row, ents, ent_ments[sid],
                                num_ents[sid], split, None,
                                same_row=same_rows)
                    visited = visited.union(same_rows.index)
                else:
                    self.do_row(row, ents, ent_ments[sid], counter,
                                num_ents[sid], split)
                bar.update(bar.value + 1)

            self.offset[split] = {i: 0 for i in self.labels[split]}
            self.unfilled[split] = set(self.offset[split].keys())

        return self

    def do_context(self, ctx_, char_lines, line_num):
        # Get lines in story where character appears
        # To trim context to right parts
        ctx = []
        for line in char_lines:
            if line < line_num:
                ctx += [self.vocabs["sentence"][i] for i in ctx_[line - 1]]
        return ctx

    def do_row(self, row, ents, ent_ments, num_ents,
               split, counter, same_row=None):
        # Find entity in entity file
        story = row["storyid"]
        char = row["char"]
        line_num = row["linenum"]

        raw_sent = lit_eval(row["sentence"])
        ctx_ = ctx_lit_eval(row["context"])

        char_lines = ents[(story, char)]

        ctx = self.do_context(ctx_, char_lines, line_num)

        # Get context and sentence
        sent = [self.vocabs["sentence"][i] for i in raw_sent]

        # Get story id_
        if same_row is not None:
            id_ = (story, char, line_num)
        else:
            counter.setdefault((story, char, line_num), 0)
            id_ = (story, char, line_num, counter[(story, char, line_num)])
            counter[(story, char, line_num)] += 1

        sl = len(sent)
        cl = len(ctx)

        sent_diff = self.sent_maxes[split] - sl
        ctx_diff = self.ctx_maxes[split].get(line_num - 2, 1) - cl

        sent += [0] * sent_diff
        ctx += [0] * ctx_diff

        labels, raw_labels = self.make_label_point(row, same_row)

        key = line_num

        if key not in self.sent[split]:
            self.sent[split][key] = \
                torch.LongTensor(sent).unsqueeze(0)
            self.ctx[split][key] = \
                torch.LongTensor(ctx).unsqueeze(0)
            self.labels[split][key] = \
                torch.FloatTensor(labels).unsqueeze(0)
            self.sent_lengths[split][key] = \
                torch.LongTensor([sl])
            self.ctx_lengths[split][key] = \
                torch.LongTensor([cl])
            self.ids[split][key] = [id_]
        else:
            self.sent[split][key] = torch.cat(
                [self.sent[split][key],
                 torch.LongTensor(sent).unsqueeze(0)], 0)
            self.ctx[split][key] = torch.cat(
                [self.ctx[split][key],
                 torch.LongTensor(ctx).unsqueeze(0)], 0)
            self.labels[split][key] = torch.cat(
                [self.labels[split][key],
                 torch.FloatTensor(labels).unsqueeze(0)], 0)
            self.sent_lengths[split][key] = torch.cat(
                [self.sent_lengths[split][key],
                 torch.LongTensor([sl])], 0)
            self.ctx_lengths[split][key] = torch.cat(
                [self.ctx_lengths[split][key],
                 torch.LongTensor([cl])], 0)
            self.ids[split][key] += [id_]
        self.raw[split][id_] = self.raw[split].get(id_, []) + [raw_sent]
        self.raw_labels[split][id_] = raw_labels
        self.size[split][key] = self.size[split].get(key, 0) + 1

    # If you only have a development set, but want to train on a part of it,
    # split the dev set into training/dev split using t_ratio % of the dev set
    # for training and the remaining for test
    def split_set(self, t_ratio=0.8, keep="dev", write="train"):
        self.labels[write] = {}
        self.sent[write] = {}
        self.ctx[write] = {}
        self.sent_lengths[write] = {}
        self.ctx_lengths[write] = {}
        self.ids[write] = {}
        self.size[write] = {}
        self.raw_labels[write] = {}
        self.raw[write] = {}

        keep_vals, write_vals = self.get_splits(keep, t_ratio)

        for key in self.size[keep]:
            if write_vals:
                self.update_sets(write_vals[key], write, keep, key)

            if keep_vals:
                self.update_sets(keep_vals[key], keep, keep, key)

        self.offset[write] = {i: 0 for i in self.labels[write]}
        self.unfilled[write] = set(self.offset[write].keys())
        self.offset[keep] = {i: 0 for i in self.labels[keep]}
        self.unfilled[keep] = set(self.offset[keep].keys())

    def update_sets(self, vals, write, keep, key):
        idxs = torch.LongTensor(vals)
        if self.is_cuda:
            idxs = idxs.cuda()
        self.labels[write][key] = \
            self.labels[keep][key].index_select(0, idxs)
        self.sent[write][key] = \
            self.sent[keep][key].index_select(0, idxs)
        self.ctx[write][key] = \
            self.ctx[keep][key].index_select(0, idxs)
        self.sent_lengths[write][key] = \
            self.sent_lengths[keep][key].index_select(0, idxs)
        self.ctx_lengths[write][key] = \
            self.ctx_lengths[keep][key].index_select(0, idxs)
        p = [self.ids[keep][key][i] for i in idxs]
        self.size[write][key] = len(idxs)
        self.raw[write].update({id_: self.raw[keep][id_] for
                                id_ in p})
        self.raw_labels[write].update({id_: self.raw_labels[keep][id_] for
                                       id_ in p})
        self.ids[write][key] = p

    def sample_keys(self, split):
        return random.sample(self.unfilled[split], 1)[0]

    def shuffle_sequences(self, split, keys):
        orders = {}
        for key in keys:
            ex_order = torch.LongTensor(
                range(self.labels[split][key].size(0)))
            if self.is_cuda:
                ex_order = ex_order.cuda()
            random.shuffle(ex_order)
            orders[key] = ex_order
            self.labels[split][key] = \
                self.labels[split][key].index_select(0, ex_order)
            self.sent[split][key] = \
                self.sent[split][key].index_select(0, ex_order)
            self.ctx[split][key] = \
                self.ctx[split][key].index_select(0, ex_order)
            self.sent_lengths[split][key] = \
                self.sent_lengths[split][key].index_select(0, ex_order)
            self.ctx_lengths[split][key] = \
                self.ctx_lengths[split][key].index_select(0, ex_order)
            self.ids[split][key] = [self.ids[split][key][i]
                                    for i in ex_order]
        return orders

    # keyss should be a list of keyss to reset
    def reset_offsets(self, split=None, shuffle=True):
        def reset_offset(split, keyss):
            for keys in keyss:
                if keys in self.offset[split]:
                    self.offset[split][keys] = 0
                else:
                    print "keys not in offset"
            self.unfilled[split] = \
                self.unfilled[split].union(keyss)

        if not split:
            splits = self.splits
        else:
            splits = [split]

        orders = {}

        for split in splits:
            keyss = self.offset[split].keys()
            if shuffle:
                orders[split] = \
                    self.shuffle_sequences(split, keyss)
            reset_offset(split, keyss)
        return orders

    def sample_batches(self, split, keys=None, bs=None):
        # If specific key isn't specified, sample a key
        if not keys:
            keys = self.sample_keys(split)

        if not bs:
            bs = self.batch_size

        offset = self.offset[split]

        # Choose sequences to batch
        start_idx = offset[keys]
        final_idx = offset[keys] + bs

        num_sentences = self.sent[split][keys].size(0)
        labels = self.labels[split][keys]
        sentence = self.sent[split][keys]
        sentence_lengths = self.sent_lengths[split][keys]
        context = self.ctx[split][keys]
        context_lengths = self.ctx_lengths[split][keys]
        ids = self.ids[split][keys]

        # Check if final index is greater than number of sequences
        if final_idx < num_sentences:
            idx_range = range(start_idx, final_idx)
            idxs = torch.LongTensor(idx_range)
            # Update offset for next access of this key
            self.offset[split][keys] += bs
        else:
            # If it is, take a subset of the self.batch_size
            idx_range = range(start_idx, num_sentences)
            idxs = torch.LongTensor(idx_range)
            self.offset[split][keys] = 0
            if keys in self.unfilled[split]:
                # This key has been visited completed
                self.unfilled[split].remove(keys)

        if self.is_cuda:
            idxs = idxs.cuda(cfg.device)

        ss = sentence.index_select(0, idxs)
        sl = sentence_lengths.index_select(0, idxs)
        cs = context.index_select(0, idxs)
        cl = context_lengths.index_select(0, idxs)
        lab = labels.index_select(0, idxs)
        idss = ids[idx_range[0]:idx_range[-1] + 1]

        return lab, ss, sl, cs, cl, None, None, None, idss, keys

    def cuda(self, device_id=None):
        if device_id is None:
            device_id = cfg.device
        for split in self.sent.keys():
            for key in self.sent[split].keys():
                self.sent[split][key] = \
                    self.sent[split][key].cuda(device_id)
                self.sent_lengths[split][key] = \
                    self.sent_lengths[split][key].cuda(device_id)
                self.ctx[split][key] = \
                    self.ctx[split][key].cuda(device_id)
                self.ctx_lengths[split][key] = \
                    self.ctx_lengths[split][key].cuda(device_id)
                self.labels[split][key] = \
                    self.labels[split][key].cuda(device_id)

        self.is_cuda = True


class MemoryModelDataLoader(NeuralModelDataLoader):
    def __init__(self, opt=None, batch_size=32):
        super(MemoryModelDataLoader, self).__init__(opt, batch_size)
        self.is_ren = True

    def do_context(self, ctx_, char_lines, line_num):
        # Place context lines for a story sentence in a dictionary
        # where each key is the order of the sentence
        ctx = {}
        for line in range(len(ctx_)):
            if line < line_num:
                ctx[line + 1] = [self.vocabs["sentence"][i]
                                 for i in ctx_[line]]
        return ctx

    # ent_ments is entities mentioned in every line
    # num_ents is the entity identity
    def do_entities(self, ent_ments, num_ents):
        elabel = {}
        eids = [0] * len(num_ents)

        # Get entity ids from vocab
        for ent, idx in num_ents.iteritems():
            if ent.lower() not in self.vocabs["entity"]._index:
                print ent.lower()
            eids[idx] = self.vocabs["entity"][ent.lower()]

        # Get entity labels at each step for
        # additional supervision
        for i in range(1, 6):
            ent_list = ent_ments.get(i, [])
            elabel[i] = [0] * len(num_ents)
            for ent in ent_list:
                elabel[i][num_ents[ent]] = 1
        return eids, elabel

    def do_row(self, row, ents, ent_ments, num_ents,
               split, counter, same_row=None):
        # Find entity in entity file
        story = row["storyid"]
        char = row["char"]
        line_num = row["linenum"]

        # Extract sentences and context
        raw_sent = lit_eval(row["sentence"])
        ctx_ = ctx_lit_eval(row["context"])

        # Find the lines in the story in which this character appears
        char_lines = ents[(story, char)]

        # Get previous sentences to make context data structure
        ctx = self.do_context(ctx_, char_lines, line_num)

        # Get entities
        eids, elabels = self.do_entities(ent_ments, num_ents)

        # Get context and sentence
        sent = [self.vocabs["sentence"][i] for i in raw_sent]

        # Get story id_
        if same_row is not None:
            id_ = (story, char, line_num)
        else:
            counter.setdefault((story, char, line_num), 0)
            id_ = (story, char, line_num, counter[(story, char, line_num)])
            counter[(story, char, line_num)] += 1

        # Get sentence lengths
        sl = len(sent)
        sent_diff = self.sent_maxes[split] - sl
        sent += [0] * sent_diff

        # Get context lengths
        clengths = {}
        for i in ctx:
            clengths[i] = len(ctx[i])
            ctx_diff = self.ctx_maxes[split][i - 1] - clengths[i]
            ctx[i] += [0] * ctx_diff

        # Get labels for this point
        labels, raw_labels = self.make_label_point(row, same_row)

        # Make the key for the group this data point belongs to
        key = (line_num, len(num_ents))

        # Add the example to all the corresponding data stores
        # for each part of the data point (sentence, labels, context, etc.)
        if key not in self.sent[split]:
            self.sent[split][key] = \
                torch.LongTensor(sent).unsqueeze(0)
            self.labels[split][key] = \
                torch.FloatTensor(labels).unsqueeze(0)
            self.sent_lengths[split][key] = \
                torch.LongTensor([sl])
            self.ent[split][key] = \
                torch.LongTensor([num_ents[char]])
            self.ent_init[split][key] = \
                torch.LongTensor(eids).unsqueeze(0)
            self.ids[split][key] = [id_]

            self.ctx[split][key] = {}
            self.ctx_lengths[split][key] = {}
            self.ent_labels[split][key] = {}
            for i in range(1, key[0]):
                self.ctx[split][key][i] = \
                    torch.LongTensor(ctx[i]).unsqueeze(0)
                self.ctx_lengths[split][key][i] = \
                    torch.LongTensor([clengths[i]])
                self.ent_labels[split][key][i] = \
                    torch.FloatTensor([elabels[i]])
            self.ent_labels[split][key][key[0]] = \
                torch.FloatTensor([elabels[key[0]]])
        else:
            self.sent[split][key] = torch.cat(
                [self.sent[split][key],
                 torch.LongTensor(sent).unsqueeze(0)], 0)
            self.labels[split][key] = torch.cat(
                [self.labels[split][key],
                 torch.FloatTensor(labels).unsqueeze(0)], 0)
            self.sent_lengths[split][key] = torch.cat(
                [self.sent_lengths[split][key],
                 torch.LongTensor([sl])], 0)
            self.ent[split][key] = torch.cat(
                [self.ent[split][key],
                 torch.LongTensor([num_ents[char]])], 0)
            self.ent_init[split][key] = torch.cat(
                [self.ent_init[split][key],
                 torch.LongTensor(eids).unsqueeze(0)], 0)
            for i in range(1, key[0]):
                # try:
                self.ctx[split][key][i] = torch.cat(
                    [self.ctx[split][key][i],
                     torch.LongTensor(ctx[i]).unsqueeze(0)], 0)
                self.ctx_lengths[split][key][i] = torch.cat(
                    [self.ctx_lengths[split][key][i],
                     torch.LongTensor([clengths[i]])], 0)
                self.ent_labels[split][key][i] = torch.cat(
                    [self.ent_labels[split][key][i],
                     torch.FloatTensor([elabels[i]])], 0)
            self.ent_labels[split][key][key[0]] = torch.cat(
                [self.ent_labels[split][key][key[0]],
                 torch.FloatTensor([elabels[key[0]]])], 0)
            self.ids[split][key] += [id_]
        self.raw[split][id_] = self.raw[split].get(id_, []) + [raw_sent]
        self.raw_labels[split][id_] = raw_labels
        self.size[split][key] = self.size[split].get(key, 0) + 1

    # Sample a batch from the data_loader
    def sample_batches(self, split, keys=None, bs=None):
        # If specific key isn't specified, sample a key
        if not keys:
            keys = self.sample_keys(split)

        if not bs:
            bs = self.batch_size

        offset = self.offset[split]

        # Choose sequences to batch
        start_idx = offset[keys]
        final_idx = offset[keys] + bs

        # Cache the correct data stores
        num_sentences = self.sent[split][keys].size(0)
        labels = self.labels[split][keys]
        sentence = self.sent[split][keys]
        sentence_lengths = self.sent_lengths[split][keys]
        context = self.ctx[split][keys]
        context_lengths = self.ctx_lengths[split][keys]
        ent = self.ent[split][keys]
        ent_init = self.ent_init[split][keys]
        ent_labels = self.ent_labels[split][keys]
        ids = self.ids[split][keys]

        # Check if final index is greater than number of sequences
        if final_idx < num_sentences:
            idx_range = range(start_idx, final_idx)
            idxs = torch.LongTensor(idx_range)
            # Update offset for next access of this key
            self.offset[split][keys] += bs
        else:
            # If it is, take a subset of the self.batch_size
            idx_range = range(start_idx, num_sentences)
            idxs = torch.LongTensor(idx_range)
            self.offset[split][keys] = 0
            if keys in self.unfilled[split]:
                # This key has been visited completed
                self.unfilled[split].remove(keys)

        if self.is_cuda:
            idxs = idxs.cuda(cfg.device)

        s = sorted(context.keys())

        # Select the correct examples using the sampled indices
        ss = sentence.index_select(0, idxs)
        sl = sentence_lengths.index_select(0, idxs)
        cs = {cn: context[cn].index_select(0, idxs) for cn in s}
        cl = {cn: context_lengths[cn].index_select(0, idxs) for cn in s}
        e = ent.index_select(0, idxs)  # entity label
        ei = ent_init.index_select(0, idxs)  # entity identities for init state
        el = {cn: ent_labels[cn].index_select(0, idxs) for cn in s}
        el[keys[0]] = ent_labels[keys[0]].index_select(0, idxs)
        lab = labels.index_select(0, idxs)
        idss = ids[idx_range[0]:idx_range[-1] + 1]

        return lab, ss, sl, cs, cl, e, ei, el, idss, keys

    # Push the data to the GPU
    def cuda(self, device_id=None):
        self.is_cuda = True
        if device_id is None:
            device_id = cfg.device
        for split in self.sent.keys():
            for key in self.sent[split].keys():
                self.sent[split][key] = \
                    self.sent[split][key].cuda(device_id)
                self.sent_lengths[split][key] = \
                    self.sent_lengths[split][key].cuda(device_id)
                self.ent[split][key] = \
                    self.ent[split][key].cuda(device_id)
                self.ent_init[split][key] = \
                    self.ent_init[split][key].cuda(device_id)
                for i in range(1, key[0]):
                    self.ctx[split][key][i] = \
                        self.ctx[split][key][i].cuda(device_id)
                    self.ctx_lengths[split][key][i] = \
                        self.ctx_lengths[split][key][i].cuda(device_id)
                    self.ent_labels[split][key][i] = \
                        self.ent_labels[split][key][i].cuda(device_id)
                self.ent_labels[split][key][key[0]] = \
                    self.ent_labels[split][key][key[0]].cuda(device_id)
                self.labels[split][key] = \
                    self.labels[split][key].cuda(device_id)

    # Shuffle all the data
    def shuffle_sequences(self, split, keys):
        orders = {}
        for key in keys:
            ex_order = torch.LongTensor(
                range(self.labels[split][key].size(0)))
            if self.is_cuda:
                ex_order = ex_order.cuda()
            random.shuffle(ex_order)
            orders[key] = ex_order
            self.labels[split][key] = \
                self.labels[split][key].index_select(0, ex_order)
            self.sent[split][key] = \
                self.sent[split][key].index_select(0, ex_order)
            self.sent_lengths[split][key] = \
                self.sent_lengths[split][key].index_select(0, ex_order)
            self.ent[split][key] = \
                self.ent[split][key].index_select(0, ex_order)
            self.ent_init[split][key] = \
                self.ent_init[split][key].index_select(0, ex_order)
            for i in range(1, key[0]):
                self.ctx[split][key][i] = \
                    self.ctx[split][key][i].index_select(0, ex_order)
                self.ctx_lengths[split][key][i] = \
                    self.ctx_lengths[split][key][i].index_select(0, ex_order)
                self.ent_labels[split][key][i] = \
                    self.ent_labels[split][key][i].index_select(0, ex_order)
            self.ent_labels[split][key][key[0]] = \
                self.ent_labels[split][key][key[0]].index_select(0, ex_order)
            self.ids[split][key] = [self.ids[split][key][i]
                                    for i in ex_order]
        return orders

    # If you only have a development set, but want to train on a part of it,
    # split the dev set into training/dev split using t_ratio % of the dev set
    # for training and the remaining for test
    def split_set(self, t_ratio=0.8, keep="dev", write="train"):
        self.labels[write] = {}
        self.sent[write] = {}
        self.ctx[write] = {}
        self.sent_lengths[write] = {}
        self.ctx_lengths[write] = {}
        self.ids[write] = {}
        self.size[write] = {}
        self.raw_labels[write] = {}
        self.raw[write] = {}
        self.ent_labels[write] = {}
        self.ent[write] = {}
        self.ent_init[write] = {}

        keep_vals, write_vals = self.get_splits(keep, t_ratio)

        for key in self.size[keep].keys():
            # print key
            if write_vals:
                self.update_sets(write_vals[key], write, keep, key)

            if keep_vals:
                self.update_sets(keep_vals[key], keep, keep, key)

        self.offset[write] = {i: 0 for i in self.labels[write]}
        self.unfilled[write] = set(self.offset[write].keys())
        self.offset[keep] = {i: 0 for i in self.labels[keep]}
        self.unfilled[keep] = set(self.offset[keep].keys())

    # Depending on the key, pop it or add it to the set of examples
    # in this split
    def update_sets(self, vals, write, keep, key):
        if not vals:
            if key in self.labels[write]:
                self.labels[write].pop(key)
                self.sent[write].pop(key)
                self.sent_lengths[write].pop(key)
                self.ent[write].pop(key)
                self.ent_init[write].pop(key)
                self.ent_labels[write].pop(key)
                self.ctx[write].pop(key)
                self.ctx_lengths[write].pop(key)
                self.size[write].pop(key)
                self.ids[write].pop(key)
        else:
            idxs = torch.LongTensor(vals)
            if self.is_cuda:
                idxs = idxs.cuda()
            self.labels[write][key] = \
                self.labels[keep][key].index_select(0, idxs)
            self.sent[write][key] = \
                self.sent[keep][key].index_select(0, idxs)
            self.sent_lengths[write][key] = \
                self.sent_lengths[keep][key].index_select(0, idxs)
            self.ent[write][key] = \
                self.ent[keep][key].index_select(0, idxs)
            self.ent_init[write][key] = \
                self.ent_init[keep][key].index_select(0, idxs)
            if key not in self.ctx[write]:
                self.ctx[write][key] = {}
                self.ctx_lengths[write][key] = {}
                self.ent_labels[write][key] = {}

            for i in range(1, key[0]):
                self.ctx[write][key][i] = \
                    self.ctx[keep][key][i].index_select(0, idxs)
                self.ctx_lengths[write][key][i] = \
                    self.ctx_lengths[keep][key][i].index_select(0, idxs)
                self.ent_labels[write][key][i] = \
                    self.ent_labels[keep][key][i].index_select(0, idxs)
            self.ent_labels[write][key][key[0]] = \
                self.ent_labels[keep][key][key[0]].index_select(0, idxs)
            p = [self.ids[keep][key][i] for i in idxs]
            self.size[write][key] = len(idxs)
            self.raw[write].update({id_: self.raw[keep][id_] for
                                    id_ in p})
            self.raw_labels[write].update({id_: self.raw_labels[keep][id_] for
                                           id_ in p})
            self.ids[write][key] = p
