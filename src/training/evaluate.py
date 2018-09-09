"""
File for methods related to evaluating model performance
on the story commonsense dataset.

author: Antoine Bosselut (atcbosselut)
"""

import time
import progressbar

import torch
from torch.autograd import Variable
import torch.optim

import src.data.config as cfg
import utils.model_utils as model_utils


def eval_gen(opt, model, data_loader, gpu=True, split="dev"):
    encoder = model.encoder
    decoder = model.decoder
    e_encoder = model.ent_encoder

    # Initialize loss function
    word_crit = model_utils.crits(opt.train.static.wcrit)
    if gpu:
        word_crit = word_crit.cuda(cfg.device)

    # Set losses to track
    average_loss = {"token_{}_pos".format(opt.task): 0,
                    "token_{}_neg".format(opt.task): 0,
                    "token_{}".format(opt.task): 0}

    num = {"pos": 0, "neg": 0}

    # Reset data offsets for evaluation set
    data_loader.reset_offsets(split=split, pn="pos", shuffle=False)
    data_loader.reset_offsets(split=split, pn="neg", shuffle=False)

    # Initialize Progress Bar
    num_examples = (sum(data_loader.size["pos"][split].values()) +
                    sum(data_loader.size["neg"][split].values()))
    bar = progressbar.ProgressBar(max_value=num_examples)
    bar.update(0)

    model.eval()

    start = time.time()

    for pn in ["pos", "neg"]:
        for length in data_loader.offset[pn][split].keys():
            while length in data_loader.unfilled[pn][split]:
                # Sample a batch from the data_loader
                _seqs, _sent, _sl, _ctx, _cl, e, ei, el, _, keys = \
                    data_loader.sample_batches(split, keys=length, pn=pn)

                bs = _seqs.size(0)
                token_loss = 0

                # Encode input
                hidden, _ = encode_input(
                    opt, bs, encoder, e_encoder, _sent,
                    _sl, _ctx, _cl, e, ei, el, keys[1])

                hidden = model.hproj(hidden)
                orig_hidden = hidden

                # Decode
                input_ = Variable(_seqs)
                for i in range(_seqs.size(1) - 1):
                    word_acts, hidden = decoder(
                        input_[:, i], hidden, orig_hidden)

                    # Set attention and choice labels
                    word_dist = model_utils.modify_output_for_crit(
                        opt.train.static.wcrit, word_acts)

                    # Compute losses
                    token_loss += word_crit(word_dist, input_[:, i + 1])

                # Update losses
                code = "token_{}_{}".format(opt.task, pn)
                to_add = (token_loss.data * bs / (_seqs.size(1) - 1))
                average_loss[code] += to_add[0]
                average_loss["token_{}".format(opt.task)] += to_add[0]
                num[pn] += bs

                bar.update(bar.value + bs)

    if gpu:
        torch.cuda.synchronize()

    print "Dev evaluation completed in: {} s".format(time.time() - start)

    # Get normalizers for each loss
    divs = {
        "token_{}".format(opt.task): num["neg"] + num["pos"],
        "token_{}_neg".format(opt.task, pn): num["neg"],
        "token_{}_pos".format(opt.task, pn): num["pos"],
    }

    return {i: j / divs[i] for i, j in average_loss.iteritems()}


def eval_class(opt, model, data_loader, gpu=True, split="dev"):
    encoder = model.encoder
    e_encoder = model.ent_encoder
    classifier = model.classifier

    # Initialize losses to track
    average_loss = {opt.granularity: 0}
    num = 0

    # Reset offset
    data_loader.reset_offsets(split=split, shuffle=False)

    model.eval()

    # Initialize scores to track
    scores, counts = initialize_scores(
        data_loader.vocabs[opt.granularity]._tokens)

    # Initialize answers to track
    answers = {}

    start = time.time()

    # Initialize progress bar
    num_examples = sum([tensor.size(0) for tensor in
                        data_loader.labels[split].values()])
    bar = progressbar.ProgressBar(max_value=num_examples)
    bar.update(0)

    for length in data_loader.offset[split].keys():
        while length in data_loader.unfilled[split]:
            # Sample batch from data_loader
            _labs, _sent, _sl, _ctx, _cl, e, ei, el, ids, keys = \
                data_loader.sample_batches(split, keys=length)

            model.zero_grad()
            bs = _labs.size(0)

            # Encode inpute
            hidden, _ = encode_input(
                opt, bs, encoder, e_encoder, _sent,
                _sl, _ctx, _cl, e, ei, el, keys)

            # Set label weights for loss function
            wts = ((1 - data_loader.class_weights[split]) * _labs +
                   data_loader.class_weights[split] * (1 - _labs))

            # Predict labels
            act, loss, prob = classifier(
                hidden, Variable(_labs), Variable(wts))

            # Record loss
            average_loss[opt.granularity] += loss.data * bs
            num += bs

            # Record answer scores
            scores, counts = compute_scores(
                data_loader.vocabs[opt.granularity]._tokens,
                scores, counts, _labs, prob.data.round())

            # Record answers
            answers = compute_answers(
                data_loader.vocabs[opt.granularity]._tokens, ids,
                answers, prob.data.round())

            # Update progress bar
            bar.update(bar.value + bs)

    if gpu:
        torch.cuda.synchronize()

    print "Dev evaluation completed in: {} s".format(time.time() - start)

    return {i: j[0] / num for i, j in average_loss.iteritems()}, \
        scores, counts, answers


def encode_input(opt, bs, encoder, e_encoder, _sent,
                 _sl, _ctx, _cl, e, ei, el, key):
    if opt.net.enc.model in ["ren", "npn"]:
        if _ctx:
            # Make sentence last part of context
            # for memory models
            sent_idx = max(_ctx.keys()) + 1
            _ctx[sent_idx] = _sent
            _cl[sent_idx] = _sl

            # Encode story up to this points
            entities, e_hidden, sel_loss = encoder(
                _ctx, _cl, el, Variable(ei))
        else:
            # Encode sentence
            entities, e_hidden, sel_loss = encoder(
                {1: _sent}, {1: _sl}, el, Variable(ei))
        if opt.net.enc.tied:
            # If using tied entities, select the memory slot for the
            # entity that we are predicting motivations or emotions for
            e_idx = e.view(-1, 1, 1).expand(
                bs, 1, entities.size(2))
            hidden = entities.gather(1, Variable(e_idx)).squeeze(1)
        else:
            # if entities, aren't tied, use the average entity
            # computed by attending to the entity memoruy using
            # the entity selection weights for this sentence
            hidden = e_hidden
    else:
        # Encode sentence
        s_hidden, contexts, sel_loss = encoder(Variable(_sent), _sl)
        if e_encoder:
            # Encode context
            if key != 1:
                e_hidden, e_ctx, _ = e_encoder(
                    Variable(_ctx), _cl)
            else:
                # If it's the first sentence, make context a zero vector
                e_hidden = Variable(s_hidden.data.clone().zero_())

            # Concatenate context and sentence vectors
            hidden = torch.cat([s_hidden, e_hidden], 1)
        else:
            hidden = s_hidden

    return hidden, sel_loss


def initialize_scores(vocab):
    scores = {}
    counts = {}

    # Individual category scores
    for i, word in vocab.iteritems():
        scores["{}_precision".format(word)] = 0
        scores["{}_recall".format(word)] = 0
        scores["{}_accuracy".format(word)] = 0

        counts["{}_precision".format(word)] = 0
        counts["{}_recall".format(word)] = 0
        counts["{}_accuracy".format(word)] = 0

    # Scores across all categories
    word = "total"
    scores["{}_precision_micro".format(word)] = 0
    scores["{}_recall_micro".format(word)] = 0
    scores["{}_precision_macro".format(word)] = 0
    scores["{}_recall_macro".format(word)] = 0
    scores["{}_accuracy".format(word)] = 0

    counts["{}_precision_micro".format(word)] = 0
    counts["{}_recall_micro".format(word)] = 0
    counts["{}_precision_macro".format(word)] = 0
    counts["{}_recall_macro".format(word)] = 0
    counts["{}_accuracy".format(word)] = 0

    return scores, counts


def compute_scores(vocab, scores, counts, _labels, predicted):
    bs = _labels.size(0)
    vocab_size = _labels.size(1)

    # Precision is looks at how many positive labels we get right
    # from the ones we predict
    local_precision = ((_labels == predicted) & predicted.byte()).float()

    # Recall counts the number of positive labels we recover
    local_recall = ((_labels == predicted) & _labels.byte()).float()

    # Accuracy just looks at whether we predict the same as the label
    # regardless of whether it's negative or positive
    correct = (_labels == predicted)

    # Compute precision, recall, accuracy for each possible label
    # e.g., for emotions this is the 8 emotions of Plutchik's wheel
    for i, word in vocab.iteritems():
        # Numerators
        scores["{}_precision".format(word)] += local_precision[:, i].sum()
        scores["{}_recall".format(word)] += local_recall[:, i].sum()
        scores["{}_accuracy".format(word)] += correct[:, i].sum()

        # Normalizers
        counts["{}_precision".format(word)] += predicted[:, i].sum()
        counts["{}_recall".format(word)] += _labels[:, i].sum()
        counts["{}_accuracy".format(word)] += bs

    word = "total"

    # Count number of predictions and labels
    prec_norm = predicted.sum(1)
    rec_norm = _labels.sum(1)

    # Add 1 if there are no positive labels or predictions
    prec_cover = (prec_norm == 0).float() + prec_norm
    rec_cover = (rec_norm == 0).float() + rec_norm

    # Compute a macroaverage over all the  labels
    scores["{}_precision_macro".format(word)] += \
        (local_precision / prec_cover.unsqueeze(1).repeat(
            1, vocab_size)).sum()
    scores["{}_recall_macro".format(word)] += \
        (local_recall / rec_cover.unsqueeze(1).repeat(
            1, vocab_size)).sum()

    # Compute a microaverage over all the labels
    # (this is the one reported in the paper)
    scores["{}_precision_micro".format(word)] += \
        local_precision.sum()
    scores["{}_recall_micro".format(word)] += \
        local_recall.sum()
    scores["{}_accuracy".format(word)] += correct.sum()

    counts["{}_precision_macro".format(word)] += (prec_norm != 0).sum()
    counts["{}_recall_macro".format(word)] += (rec_norm != 0).sum()
    counts["{}_precision_micro".format(word)] += predicted.sum()
    counts["{}_recall_micro".format(word)] += _labels.sum()
    counts["{}_accuracy".format(word)] += (bs * vocab_size)

    return scores, counts


def compute_answers(vocab, ids, answers, predicted):
    # Convert predictions to vocabulary for saving answers
    for id_idx, id_ in enumerate(ids):
        answers[id_] = [vocab[i] for i, j in enumerate(
            predicted[id_idx].tolist()) if j]
    return answers


eval_function = {
    "gen": eval_gen,
    "class": eval_class
}
