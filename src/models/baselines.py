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
    def __init__(self, opt, vocab, ent_vocab=None, max_words=None):
        super(Model, self).__init__()

        self.mult = 1

        self.opt = opt

        # Initialize sentence encoder
        if opt.enc.model == "ren":
            self.encoder = entnet.EntNet(opt.enc, max_words)
            if "gauss" in opt.enc.init:
                deets = opt.enc.init.split("+")
                mean = deets[1]
                stddev = deets[2]
                print("Using Gaussian Initialization with Mean = {} and \
                    Standard Deviation = {}").format(
                    float(mean), float(stddev))
                for parameter in self.encoder.parameters():
                    parameter.data.normal_(float(mean), float(stddev))
            self.encoder.encoder.sentence_encoder.weights.data.fill_(1)
            if opt.enc.pt != "none":
                self.encoder.encoder.sentence_encoder.load_embeddings(vocab)
            if opt.enc.entpt != "none":
                self.encoder.load_entity_embeddings(ent_vocab)

        elif opt.enc.model == "npn":
            self.encoder = npn.NPN(opt.enc, max_words)
            if "gauss" in opt.enc.init:
                deets = opt.enc.init.split("+")
                mean = deets[1]
                stddev = deets[2]
                print("Using Gaussian Initialization with Mean = {} and \
                    Standard Deviation = {}").format(
                    float(mean), float(stddev))
                for parameter in self.encoder.parameters():
                    parameter.data.normal_(float(mean), float(stddev))
            self.encoder.encoder.sentence_encoder.weights.data.fill_(1)
            if opt.enc.pt != "none":
                self.encoder.encoder.sentence_encoder.load_embeddings(vocab)
                self.encoder.action_enc.sentence_encoder.load_embeddings(vocab)
            if opt.enc.entpt != "none":
                self.encoder.load_entity_embeddings(ent_vocab)
            self.encoder.initialize_actions(opt.enc.aI, ent_vocab)

        elif opt.enc.model in ["rnn", "lstm", "gru"]:
            self.encoder = RecurrentEncoder(opt.enc)
            if opt.enc.pt != "none":
                self.encoder.sentence_encoder.load_embeddings(vocab)
            self.mult += opt.enc.get("bid", 0)
            self.mult *= opt.enc.get("nL", 1)

        elif opt.enc.model == "cnn":
            self.encoder = CNNEncoder(opt.enc)
            if opt.enc.pt != "none":
                self.encoder.load_embeddings(vocab)
            self.mult = len(opt.enc.ks.split(","))

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
    def __init__(self, opt, vocab, ent_vocab=None,
                 max_words=None, crit_weights=None):
        super(ClassModel, self).__init__(
            opt, vocab, ent_vocab, max_words)

        self.classifier = StatePredictor(
            opt.classdec, mult=self.mult,
            do_ents=(opt.enc.ctx and opt.enc.model not in ["ren", "npn"]))

    def forward(self):
        pass

    def cuda(self, device_id):
        super(ClassModel, self).cuda(device_id)
        self.classifier.cuda(device_id)


class GenModel(Model):
    def __init__(self, opt, vocab, ent_vocab=None, max_words=None):
        super(GenModel, self).__init__(opt, vocab, ent_vocab, max_words)

        condition = (opt.enc.ctx and opt.enc.model not in ["ren", "npn"])

        mult = self.mult * (1 + condition)

        self.hproj = nn.Linear(
            mult * opt.gendec.hSize, opt.gendec.hSize)

        # Initialize decoder
        if opt.gendec.model == "aed":
            print "Attentive Decoder not implemented"
            raise
            # self.decoder = models.AttentiveDecoder(opt.gendec, mult=mult)
        else:
            self.decoder = models.Decoder(opt.gendec)

    def cuda(self, device_id):
        super(GenModel, self).cuda(device_id)
        self.decoder.cuda(device_id)


class RecurrentEncoder(nn.Module):
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
class CNNEncoder(nn.Module):
    def __init__(self, opt):
        super(CNNEncoder, self).__init__()
        self.opt = opt
        self.is_cuda = False

        # V = args.embed_num #
        # D = args.embed_dim #
        # C = opt.class_num #
        Ci = 1
        Co = opt.kn
        Ks = opt.ks
        print opt.iSize
        self.embs = nn.Embedding(opt.vSize, opt.iSize)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks] #
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (int(K), opt.iSize))
                                     for K in Ks.split(",")])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(opt.dpt)
        # self.fc1 = nn.Linear(len(Ks)*Co, C) #

    # def conv_and_pool(self, x, conv):
    #     x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
    #     x = F.max_pool1d(x, x.size(2)).squeeze(2)
    #     return x

    def forward(self, x, lengths):
        x = self.embs(x)  # (N, W, D)
        x = self.dropout(x)
        # if self.args.static:
        #     x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        # [(N, Co, W), ...]*len(Ks)
        # [(N, Co), ...]*len(Ks)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        # x = self.dropout(x)  # (N, len(Ks)*Co)
        # logit = self.fc1(x)  # (N, C)
        return x, None, None

    def load_embeddings(self, vocab):
        if self.opt.pt != "none":
            name = cfg.get_glove_name(self.opt, "tokens")
            print "Loading embeddings from {}".format(name)
            with open(name, "r") as f:
                token_words = pickle.load(f)
            # print vocab
            for i, word in vocab.iteritems():
                if word in ["<unk>", "<start>", "<end>", "<pad>"]:
                    self.embs.weight.data[i].zero_()
                    continue
                if self.is_cuda:
                    vec = torch.cuda.FloatTensor(token_words[word])
                else:
                    vec = torch.FloatTensor(token_words[word])
                self.embs.weight.data[i] = vec

    def cuda(self, device_id):
        super(CNNEncoder, self).cuda(device_id)
        self.embs.cuda(device_id)
        self.convs1.cuda(device_id)
        self.is_cuda = True


class StatePredictor(nn.Module):
    def __init__(self, opt, mult=1, do_ents=False):
        super(StatePredictor, self).__init__()

        self.dropout = nn.Dropout(opt.dpt)
        self.proj = nn.Linear(mult * (1 + do_ents) * opt.hSize, opt.vSize)
        self.prob = nn.Sigmoid()

    def forward(self, input, labels, weights):
        # print(input.size())
        # print(self.proj)
        act = self.proj(self.dropout(input))

        loss = F.binary_cross_entropy_with_logits(
            act, labels, weight=weights)
        # loss = self.loss_fn(act, labels)

        prob = self.prob(Variable(act.data, volatile=True))
        return act, loss, prob

    def cuda(self, device_id):
        super(StatePredictor, self).cuda(device_id)
