"""
Library of certain baselines used in the paper

author: Antoine Bosselut (atcbosselut)
"""

import src.data.config as cfg
import src.models.models as models
import src.models.npn as npn
import src.models.entnet as entnet

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import pickle


class Model(nn.Module):
    """
    Container class for holding an encoder and some type of classifier
    or decoder

    Initialization Args:
        opt: options to pass to encoder and classifier/decoder
        vocab: vocabulary for the encoder
        ent_vocab: vocabulary of possible entities
        max_words: maximum number of words in a sentence
                   (useful for the positional encoder of EntNet)
    """
    def __init__(self, opt, vocab, ent_vocab=None, max_words=None):
        super(Model, self).__init__()

        self.mult = 1

        self.opt = opt

        # Initialize sentence encoder
        if opt.enc.model == "ren":
            self.encoder = entnet.EntNet(opt.enc, max_words)
            if "gauss" in opt.enc.init:
                mean_std = opt.enc.init.split("+")
                mean = mean_std[1]
                stddev = mean_std[2]
                print("Using Gaussian Initialization with Mean = {} and \
                    Standard Deviation = {}").format(
                    float(mean), float(stddev))
                for parameter in self.encoder.parameters():
                    parameter.data.normal_(float(mean), float(stddev))
            # Initialize Weighted BOW weights as 1
            # similar to Henaff et al., 2016
            self.encoder.encoder.sentence_encoder.weights.data.fill_(1)

            # Load pretrained embeddings
            if opt.enc.pt != "none":
                self.encoder.encoder.sentence_encoder.load_embeddings(vocab)
            if opt.enc.entpt != "none":
                self.encoder.load_entity_embeddings(ent_vocab)

        # This version of the NPN uses a similar sentence encoder as EntNet
        elif opt.enc.model == "npn":
            self.encoder = npn.NPN(opt.enc, max_words)
            if "gauss" in opt.enc.init:
                mean_std = opt.enc.init.split("+")
                mean = mean_std[1]
                stddev = mean_std[2]
                print("Using Gaussian Initialization with Mean = {} and \
                    Standard Deviation = {}").format(
                    float(mean), float(stddev))
                for parameter in self.encoder.parameters():
                    parameter.data.normal_(float(mean), float(stddev))
            self.encoder.encoder.sentence_encoder.weights.data.fill_(1)

            # Load pretrained embeddings
            if opt.enc.pt != "none":
                self.encoder.encoder.sentence_encoder.load_embeddings(vocab)
                self.encoder.action_enc.sentence_encoder.load_embeddings(vocab)
            if opt.enc.entpt != "none":
                self.encoder.load_entity_embeddings(ent_vocab)

            # Initialize action functions
            self.encoder.initialize_actions(opt.enc.aI, ent_vocab)

        elif opt.enc.model in ["rnn", "lstm", "gru"]:
            self.encoder = RecurrentEncoder(opt.enc)

            # Load pretrained embeddings
            if opt.enc.pt != "none":
                self.encoder.sentence_encoder.load_embeddings(vocab)

            self.mult += opt.enc.get("bid", 0)
            self.mult *= opt.enc.get("nL", 1)

        elif opt.enc.model == "cnn":
            self.encoder = CNNEncoder(opt.enc)

            # Load pretrained embeddings
            if opt.enc.pt != "none":
                self.encoder.load_embeddings(vocab)

            # Depending on how many different kernel sizes
            # we're using, change the size of the projection
            # before passing to decoder/classifier
            self.mult = len(opt.enc.ks.split(","))

        # If we're using entity-specific context for a neural model,
        # set a different encoder with the same architecture as for the
        # sentence encoder
        if opt.enc.ctx:
            if opt.enc.model in ["rnn", "lstm", "gru"]:
                self.ent_encoder = RecurrentEncoder(opt.enc)
                if opt.enc.pt != "none":
                    self.ent_encoder.sentence_encoder.load_embeddings(vocab)
            elif opt.enc.model == "cnn":
                self.ent_encoder = CNNEncoder(opt.enc)
                if opt.enc.pt != "none":
                    self.ent_encoder.load_embeddings(vocab)
            else:
                self.ent_encoder = None
        else:
            self.ent_encoder = None

    def forward(self):
        pass

    def cuda(self, device_id):
        super(Model, self).cuda(device_id)
        self.encoder.cuda(device_id)
        if self.opt.enc.ctx and self.opt.enc.model not in ["ren", "npn"]:
            self.ent_encoder.cuda(device_id)


class ClassModel(Model):
    """
    Container class for holding an encoder and a classifier

    Initialization Args:
        opt: options to pass to encoder and classifier
        vocab: vocabulary for the encoder
        ent_vocab: vocabulary of possible entities
        max_words: maximum number of words in a sentence
                   (useful for the positional encoder of EntNet)
    """
    def __init__(self, opt, vocab, ent_vocab=None,
                 max_words=None):
        super(ClassModel, self).__init__(
            opt, vocab, ent_vocab, max_words)

        # condition is whether to increase the size of the linear
        # transformation to the vocabulary space because we're including
        # entity-specific context information for a neural model such as
        # CNN or LSTM.
        # Refer to the paper for more details.
        condition = (opt.enc.ctx and opt.enc.model not in ["ren", "npn"])

        # Initialize classifier
        self.classifier = StatePredictor(
            opt.classdec, mult=self.mult,
            ctx=condition)

    def forward(self):
        pass

    def cuda(self, device_id):
        super(ClassModel, self).cuda(device_id)
        self.classifier.cuda(device_id)


class GenModel(Model):
    """
    Container class for holding an encoder and a decoder

    Initialization Args:
        opt: options to pass to encoder and decoder
        vocab: vocabulary for the encoder
        ent_vocab: vocabulary of possible entities
        max_words: maximum number of words in a sentence
                   (useful for the positional encoder of EntNet)
    """
    def __init__(self, opt, vocab, ent_vocab=None, max_words=None):
        super(GenModel, self).__init__(opt, vocab, ent_vocab, max_words)

        # condition is whether to increase the size of the linear
        # transformation to the vocabulary space because we're including
        # entity-specific context information for a neural model such as
        # CNN or LSTM.
        # Refer to the paper for more details.
        condition = (opt.enc.ctx and opt.enc.model not in ["ren", "npn"])
        mult = self.mult * (1 + condition)

        # Project final hidden state to decoder hidden size
        self.hproj = nn.Linear(
            mult * opt.gendec.hSize, opt.gendec.hSize)

        # Initialize decoder
        if opt.gendec.model == "ed":
            self.decoder = models.Decoder(opt.gendec)
        else:
            raise

    def cuda(self, device_id):
        super(GenModel, self).cuda(device_id)
        self.decoder.cuda(device_id)


class RecurrentEncoder(nn.Module):
    """
    Recurrent neural network encoder

    Initialization Args:
        opt: options to pass to encoder and decoder

    Inputs:
        input: batched sequence of word indices
        lengths: length of non-pad indices in word sequences

    Outputs:
        output: output vectors from encoding words
        hidden: final layer hidden state
    """
    def __init__(self, opt):
        super(RecurrentEncoder, self).__init__()
        self.sentence_encoder = models.PaddedEncoder(opt)

        self.is_cuda = False

    def forward(self, input, lengths):
        new_lengths = lengths + (lengths == 0).long()
        output, hidden = self.sentence_encoder(input, new_lengths)
        return output, hidden, None

    def cuda(self, device_id):
        super(RecurrentEncoder, self).cuda(device_id)
        self.sentence_encoder.cuda(device_id)
        self.is_cuda = True


# Adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch
class CNNEncoder(models.PretrainedEmbeddingsModel):
    """
    Convolutional neural network encoder

    Initialization Args:
        opt: options to pass to encoder and decoder
        opt.kn: number of kernels for each size
        opt.ks: kernel sizes

    Inputs:
        x: batched sequence of word indices
        lengths: We don't need lengths here, so this is just for
                 encoder parity

    Outputs:
        output: output vectors from encoding words
        hidden: final layer hidden state
    """
    def __init__(self, opt):
        super(CNNEncoder, self).__init__()
        self.opt = opt
        self.is_cuda = False

        Ci = 1
        Co = opt.kn
        Ks = opt.ks
        print opt.iSize
        self.embeddings = nn.Embedding(opt.vSize, opt.iSize)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (int(K), opt.iSize))
                                     for K in Ks.split(",")])

        self.dropout = nn.Dropout(opt.dpt)

    def forward(self, x, lengths=None):
        x = self.embeddings(x)  # (N, W, D)
        x = self.dropout(x)

        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        return x, None, None

    def cuda(self, device_id):
        super(CNNEncoder, self).cuda(device_id)
        self.embeddings.cuda(device_id)
        self.convs1.cuda(device_id)
        self.is_cuda = True


class StatePredictor(nn.Module):
    """
    Recurrent neural network encoder

    Initialization Args:
        opt: classifier options
        mult: multiplier dictating what factor of hidden size the incoming
              input will be (e.g., if input encoder was bidirectional LSTM,
              mult would be 2)
        ctx: whether entity-specific context encoded and concatenated with
             sentence encoding

    Inputs:
        input: encoded representation of sequence (and entity)
        labels: target labels being predicted
        weights: importance of each label in the loss function

    Outputs:
        act: pre-sigmoid activation for each label
        loss: loss of predicting each label
        prob: probability of each label
    """
    def __init__(self, opt, mult=1, ctx=False):
        super(StatePredictor, self).__init__()

        self.dropout = nn.Dropout(opt.dpt)
        self.proj = nn.Linear(mult * (1 + ctx) * opt.hSize, opt.vSize)
        self.prob = nn.Sigmoid()

    def forward(self, input, labels, weights):
        act = self.proj(self.dropout(input))

        loss = F.binary_cross_entropy_with_logits(
            act, labels, weight=weights)

        prob = self.prob(Variable(act.data, volatile=True))
        return act, loss, prob

    def cuda(self, device_id):
        super(StatePredictor, self).cuda(device_id)
