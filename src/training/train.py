"""
File for methods related to training models on the
story commonsense dataset.

author: Antoine Bosselut (atcbosselut)
"""

import pickle

import torch
from torch.autograd import Variable
import torch.optim

import progressbar

import src.data.config as cfg
import src.data.data as data
import src.training.evaluate as evaluate
import utils.utils as utils
import utils.model_utils as model_utils


def train_gen(opt, meta, model, data_loader, gpu=True):
    bar = progressbar.ProgressBar()
    eval_losses = {}

    encoder = model.encoder
    e_encoder = model.ent_encoder
    decoder = model.decoder

    model.train()

    word_crit = model_utils.crits(opt.train.static.wcrit)

    if gpu:
        word_crit = word_crit.cuda(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.train.dynamic.lr)

    opt.train.dynamic.epoch = 0

    # Initialize loss trackers
    iter_loss = {"token_{}_pos".format(opt.task): [],
                 "token_{}_neg".format(opt.task): []}

    avg_loss = {"token_{}_pos".format(opt.task): 0,
                "token_{}_neg".format(opt.task): 0}

    num = {"token_{}_pos".format(opt.task): 0,
           "token_{}_neg".format(opt.task): 0}

    epoch = 0
    pns = ["neg"] + ["pos"] * opt.train.static.pnr

    for l in bar(range(1, meta.iterations)):
        if not data_loader.unfilled["neg"]["train"]:
            print "Resetting negative example offsets"
            data_loader.reset_offsets(split="train", pn="neg")

        if not l % meta.mark or not data_loader.unfilled["pos"]["train"]:
            # Do evaluation run
            eval_losses = do_gen_eval_run(
                data_loader, opt, model, gpu, eval_losses, l, split="dev")
            opt.train.dynamic.epoch = l

            data.save_step(model, optimizer, opt, l)
            data.save_train_losses(opt, iter_loss, ln="gen_losses")
            data.save_eval_losses(opt, eval_losses, "dev", ln="gen_losses")

            model.train()

            # Reset data for next epoch
            if not data_loader.unfilled["pos"]["train"]:
                print "Resetting positive example offsets"
                data_loader.reset_offsets(
                    split="train", pn="pos")
                epoch += 1

            # Reset losses
            avg_loss = {"token_{}_pos".format(opt.task): 0,
                        "token_{}_neg".format(opt.task): 0}

            num = {"token_{}_pos".format(opt.task): 0,
                   "token_{}_neg".format(opt.task): 0}

        # Based on iterations, choose
        # negative or positive sequence
        pn = pns[l % (opt.train.static.pnr + 1)]

        model.zero_grad()
        token_loss = 0

        # Sample batch
        _seqs, _sent, _sl, _ctx, _cl, e, ei, el, _, keys = \
            data_loader.sample_batches("train", pn)
        bs = _seqs.size(0)

        hidden, sel_loss = evaluate.encode_input(
            opt, bs, encoder, e_encoder, _sent,
            _sl, _ctx, _cl, e, ei, el, keys[1])

        hidden = model.hproj(hidden)
        orig_hidden = hidden

        # Decode sequence
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
        avg_loss[code] += (token_loss.data * bs / (_seqs.size(1) - 1))
        num[code] += _seqs.size(0)
        iter_loss[code].append(avg_loss[code][0] / num[code])

        # Backpropagate
        token_loss.backward()

        # Update Parameters
        torch.nn.utils.clip_grad_norm(
            model.parameters(), opt.train.static.gc)
        optimizer.step()

        opt.train.dynamic.epoch = l

    # Run evaluate script
    eval_losses = do_gen_eval_run(
        data_loader, opt, model, gpu, eval_losses, l,
        split="dev")

    data.save_step(model, optimizer, opt, l)
    data.save_train_losses(opt, iter_loss, ln="gen_losses")
    data.save_eval_losses(opt, eval_losses, "dev", ln="gen_losses")

    return iter_loss


def do_gen_eval_run(data_loader, opt, model, gpu, eval_losses,
                    l, split="dev", fn="gen"):
    # Run model on development set
    print("Evaluating {}".format(split))
    dev_loss = evaluate.eval_function[fn](
        opt, model, data_loader, gpu, "dev")

    # Report loss to stdout
    print("{} Loss: \t {}".format(
        split, dev_loss["token_{}".format(opt.task)]))
    print("Positive {} Loss: \t {}".format(
        split, dev_loss["token_{}_pos".format(opt.task)]))
    print("Negative {} Loss: \t {}".format(
        split, dev_loss["token_{}_neg".format(opt.task)]))
    eval_losses[l] = dev_loss

    return eval_losses


def do_class_eval_run(data_loader, opt, model, gpu, eval_losses, l,
                      split="dev", fn="class"):
    # Run model on development set
    print("Evaluating {}".format(split.capitalize()))
    dev_loss, scores, counts, answers = evaluate.eval_function[fn](
        opt, model, data_loader, gpu, split)

    # Print loss to stdout
    print("{} Loss: \t {}".format(
        split.capitalize(), dev_loss[opt.granularity]))

    # Record loss
    eval_losses[l] = dev_loss

    # Save Recall, Precision, F1 scores computed
    name = "{}/{}.scores".format(utils.make_name(
        opt, prefix="results/scores/", is_dir=True, eval_=True), split)
    with open(name, "w") as f:
        pickle.dump({"scores": scores, "counts": counts}, f)

    # Save individual answers
    name = "{}/{}.scores".format(utils.make_name(
        opt, prefix="results/answers/", is_dir=True, eval_=True), split)
    with open(name, "w") as f:
        pickle.dump(answers, f)

    return eval_losses, scores, counts


def train_class(opt, meta, model, data_loader, gpu=True,
                split="train", eval_split="dev"):
    eval_losses = {}

    encoder = model.encoder
    e_encoder = model.ent_encoder
    classifier = model.classifier

    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.train.dynamic.lr)

    opt.train.dynamic.epoch = 0

    top_score = 0

    for epoch in range(meta.epochs):
        model.train()
        opt.train.dynamic.epoch += 1

        # Reset losses
        iter_loss = []
        total_loss = 0
        num = 0

        # Set progress bar
        num_examples = sum([tensor.size(0) for tensor in
                            data_loader.labels["train"].values()])
        bar = progressbar.ProgressBar(max_value=num_examples)
        bar.update(0)

        while data_loader.unfilled["train"]:
            model.zero_grad()

            # Sample a batch from the data_loader
            _labs, _sent, _sl, _ctx, _cl, e, ei, el, _, keys = \
                data_loader.sample_batches(split)
            bs = _labs.size(0)

            hidden, sel_loss = evaluate.encode_input(
                opt, bs, encoder, e_encoder, _sent,
                _sl, _ctx, _cl, e, ei, el, keys)

            # Set label weights for loss computation
            wts = ((1 - data_loader.class_weights[split]) * _labs +
                   data_loader.class_weights[split] * (1 - _labs))

            # Classify labels
            act, loss, prob = classifier(
                hidden, Variable(_labs), Variable(wts))

            # Add loss to tracked loss
            total_loss += loss.data[0] * bs
            num += bs
            iter_loss.append(total_loss / num)

            # Backpropagate
            loss.backward()

            # Update parameters
            torch.nn.utils.clip_grad_norm(
                model.parameters(), opt.train.static.gc)
            optimizer.step()

            # Update progress bar
            bar.update(bar.value + bs)

        # Run evaluation mode on development set
        eval_losses, scores, counts = do_class_eval_run(
            data_loader, opt, model, gpu, eval_losses,
            opt.train.dynamic.epoch, split=eval_split)

        # Compute F1 score
        p = (1.0 * scores["total_precision_micro"] /
             max(counts["total_precision_micro"], 1))
        r = (1.0 * scores["total_recall_micro"] /
             max(counts["total_recall_micro"], 1))
        f1 = 2 * p * r / max(p + r, 1)

        # If score is improved, save model
        if f1 > top_score:
            top_score = f1
            data.save_step(model, optimizer, opt, opt.train.dynamic.epoch)

        # Save losses
        data.save_train_losses(opt, iter_loss, ln="losses")
        data.save_eval_losses(opt, eval_losses, split=eval_split,
                              ln="losses")

        data_loader.reset_offsets(split)

    return iter_loss


train_function = {
    "gen": train_gen,
    "class": train_class
}
