"""
Library of simple model classes that are often used
in text understanding

author: Antoine Bosselut (atcbosselut)
"""

import src.data.config as cfg

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import copy
import pickle


class PretrainedEmbeddingsModel(nn.Module):
    """
    Base class that allows pretrained embedding to be loaded from
    a vocabulary. Assumes the wrapper class will initialize the embeddings
    """
    def load_embeddings(self, vocab):
        if self.opt.pt != "none":
            # Get glove vector file name
            name = cfg.get_glove_name(self.opt, "tokens")
            print "Loading embeddings from {}".format(name)

            # Load glove vectors
            with open(name, "r") as f:
                token_words = pickle.load(f)

            # Assign glove vectors to correct word in vocab
            for i, word in vocab.iteritems():

                # If vocab word is meaning less, set embedding to 0
                if word in ["<unk>", "<start>", "<end>", "<pad>"]:
                    self.embeddings.weight.data[i].zero_()
                    continue

                if self.is_cuda:
                    vec = torch.cuda.FloatTensor(token_words[word])
                else:
                    vec = torch.FloatTensor(token_words[word])

                # Set embedding in embedding module
                self.embeddings.weight.data[i] = vec


class WeightedBOW(PretrainedEmbeddingsModel):
    """
    Indexes a set of word embeddings for a sequence of words it receives
    as input. Weighs each of these embeddings by a learned mask
    and returns the sum

    Initialization Args:
        opt.vSize: number of embeddings to initialize
        opt.hSize: size of embeddings to initialize
        opt.dpt: dropout probability after the embedding layer (default = 0)
        max_size: maximum number of masking functions

    Input:
        input: 3-dimensional tensor of size batch_size x seq_len

    Output:
        2-dimensional tensor of size batch_size x embed_size

    """
    def __init__(self, opt, max_size):
        super(WeightedBOW, self).__init__()
        self.embeddings = nn.Embedding(opt.vSize, opt.hSize, padding_idx=0)

        self.weights = nn.Parameter(
            torch.FloatTensor(max_size, opt.hSize).fill_(1))

        self.dropout = nn.Dropout(opt.dpt)

        self.is_cuda = False
        self.max_size = max_size
        self.opt = opt

    def forward(self, input, lengths):
        batch_size = input.size(0)
        length = input.size(1)

        embed = self.embeddings(input)
        dropped = self.dropout(embed)

        # Multiply dropped embeddings by mask weights
        w_dropped = dropped * self.weights[:length, :].unsqueeze(0).repeat(
            batch_size, 1, 1)

        return torch.sum(w_dropped, 1).view(batch_size, self.opt.hSize), None

    def cuda(self, device_id):
        super(WeightedBOW, self).cuda(device_id)
        self.weights.cuda(device_id)
        self.is_cuda = True


class PaddedRNN(nn.Module):
    """
    Indexes a set of word embeddings for a sequence of words it receives
    as input. Weighs each of these embeddings by a learned mask
    and returns the sum

    Initialization Args:
        opt.hSize: size of hidden state of GRU
        opt.nL: number of GRU layers
        opt.dpt: dropout probability between GRU layers (default = 0)
        opt.bid: use bidirectional GRU

    Input:
        input: 3-dimensional tensor of size batch_size x seq_len

    Output:
        2-dimensional tensor of size batch_size x embed_size

    """
    def __init__(self, opt):
        super(PaddedRNN, self).__init__()

        if opt.model == "gru":
            self.RNN = nn.GRU(
                opt.hSize, opt.hSize, bidirectional=opt.bid,
                dropout=opt.dpt, num_layers=opt.nL)
        elif opt.model == "lstm":
            self.RNN = nn.LSTM(
                opt.hSize, opt.hSize, bidirectional=opt.bid,
                dropout=opt.dpt, num_layers=opt.nL)
        else:
            self.RNN = nn.RNN(
                opt.hSize, opt.hSize, bidirectional=opt.bid,
                dropout=opt.dpt, num_layers=opt.nL)

        self.opt = copy.deepcopy(opt)

        self.is_cuda = False

    def forward(self, embeds, length_tensor, hidden=None):
        lengths = length_tensor.tolist()

        # Make key to recover correct batch order
        key, reverse = self.make_length_key(lengths)

        # Pack padded sequences
        padded = rnn_utils.pack_padded_sequence(
            embeds.index_select(0, key),
            sorted(lengths, reverse=True),
            batch_first=True)

        # Pass through GRU unit
        if hidden:
            out, hidden = self.RNN(padded, hidden)
        else:
            out, hidden = self.RNN(padded)

        # Unpack sequences
        padded_out, _ = rnn_utils.pad_packed_sequence(
            out, batch_first=False)

        # Recover correct batch order and return
        if self.opt.model == "lstm":
            to_cat = [hidden[0][i] for i in range(hidden[0].size(0))]
            return (torch.cat(to_cat, 1).index_select(
                0, reverse), padded_out.index_select(1, reverse))
        else:
            return (torch.cat(hidden, 1).index_select(0, reverse),
                    padded_out.index_select(1, reverse))

    def make_length_key(self, lengths):
        # If you're using Python 3, this won't work correctly...
        key = torch.LongTensor(
            [i for i, j in sorted([(i, j) for i, j in enumerate(lengths)],
                                  key=lambda x: x[1], reverse=True)])
        reverse_key = torch.LongTensor(
            {j: i for i, j in enumerate(key)}.values())
        if not self.is_cuda:
            return Variable(key), Variable(reverse_key)
        else:
            return Variable(key.cuda(cfg.device)), \
                Variable(reverse_key.cuda(cfg.device))

    def cuda(self, device_id):
        super(PaddedRNN, self).cuda(device_id)
        self.RNN.cuda(device_id)
        self.is_cuda = True


class BilinearAttention(nn.Module):
    """
    Compute a bilinear attention between two vectors

    Initialization Args:
        act: which non-linearity to use to compute attention
            (sigmoid for independent attention,
             softmax for joint attention)
        input_size_1: size of vectors to compare to
        input_size_2: size of hidden states to compare
        output_size: number of probabilities to compute
        bias: bias the bilinear projection
        temperature: temperature parameter for attention

    Input:
        input: 3-dimensional tensor of size
                batch_size x seq_len x hidden_state
                These are the hidden states to compare to
        hidden: vector used to compute attentions

    Output:
        attention distribution over hidden states
        activations for each hidden state

    """
    def __init__(self, input_size_1, input_size_2, output_size,
                 bias=True, temperature=1, act="softmax"):
        super(BilinearAttention, self).__init__()
        self.bilinear = nn.Bilinear(
            input_size_1, input_size_2, output_size, bias)

        if act == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, input, hidden):
        batch_size = input.size(0)
        num_items = input.size(1)
        input_size = input.size(2)
        hidden_size = hidden.size(1)

        hidden_ = hidden.view(batch_size, 1, hidden_size)
        hidden_ = hidden_.expand(batch_size, num_items, hidden_size)
        hidden_ = hidden_.contiguous().view(-1, hidden_size)

        inputs = input.view(-1, input_size)

        activation = self.bilinear(
            inputs, hidden_).view(batch_size, num_items)
        return self.act(activation), activation


class Attender(nn.Module):
    """
    Attend to a set of vectors using an attention distribution

    Input:
        input: 3-dimensional tensor of size
                batch_size x seq_len x hidden_state
                These are the hidden states to compare to
        weights: attention distribuution
        do_reduce: indicator variable indicating whether we compute
                   the weighted average of the attended vectors OR
                   just return them scaled by their attention weight

    Output:
        attention distribution over hidden states
        activations for each hidden state

    """
    def __init__(self):
        super(Attender, self).__init__()

    # input is batch_size * num_agenda * input_embed
    # weights is batch_size * num_agenda
    def forward(self, input, weights, do_reduce=True):
        if do_reduce:
            out = torch.bmm(weights.unsqueeze(1), input)
            return out.view(out.size(0), out.size(2))
        else:
            out = weights.unsqueeze(2).repeat(1, 1, input.size(2)) * input
            return out


class PaddedEncoder(PretrainedEmbeddingsModel):
    """
    Encode a sentence using a recurrent neural network

    Input:
        input: 2-dimensional tensor of size
                batch_size x max_seq_len
        length_tensor: lengths of the sequences in the batch

    Output:
        state -- final hidden state of each sequence
        outs -- outputs of encoder at each step

    """
    def __init__(self, opt):
        super(PaddedEncoder, self).__init__()
        self.embeddings = nn.Embedding(opt.vSize, opt.hSize, padding_idx=0)

        self.unit = PaddedRNN(opt)

        self.opt = copy.deepcopy(opt)
        self.is_cuda = False

    def forward(self, input, length_tensor):
        embeds = self.embeddings(input)
        # Get goal
        state, outs = self.unit(embeds, length_tensor)

        return state, outs

    def cuda(self, device_id):
        super(PaddedEncoder, self).cuda(device_id)
        self.embeddings.cuda(device_id)
        self.unit.cuda(device_id)
        self.is_cuda = True


class Decoder(PretrainedEmbeddingsModel):
    def __init__(self, opt, mult=1):
        super(Decoder, self).__init__()
        self.opt = opt
        self.embeddings = nn.Embedding(opt.vSize, opt.hSize)
        self.output_temperature = opt.dt

        # If we concatenate context to input, increase hidden state
        # if recurrent unit to accommodate larger input to it
        if self.opt.ctx == "inp":
            mult *= 2
            self.output_proj = nn.Linear(mult * opt.hSize, opt.vSize)

        # If we concatenate context vector to output of recurrent unit,
        # increase the size of the hidden state projection to vocabulary space
        elif self.opt.ctx == "out":
            self.output_proj = nn.Linear(2 * mult * opt.hSize, opt.vSize)

        if opt.unit == "gru":
            self.unit = nn.GRU(
                opt.hSize, mult * opt.hSize, dropout=opt.dpt)
        elif opt.unit == "lstm":
            self.unit = nn.LSTM(
                opt.hSize, mult * opt.hSize, dropout=opt.dpt)
        elif opt.unit == "rnn":
            self.unit = nn.RNN(
                opt.hSize, mult * opt.hSize, dropout=opt.dpt)

        self.dropout = nn.Dropout(opt.dpt)
        self.is_cuda = False

    def forward(self, input, hidden, goal):
        word_embs = self.embeddings(input)

        # Put attention effect at input
        if self.opt.ctx == "inp":
            output = torch.cat([word_embs, goal], 1)
        else:
            output = word_embs

        output = self.dropout(output).unsqueeze(0)
        hidden = hidden.unsqueeze(0)

        if self.opt.unit == "lstm":
            hidden = (hidden, Variable(hidden.data.clone().zero_()))

        # Run the GRU over the input and encoder hidden state
        for i in range(self.opt.nL):
            output, hidden = self.unit(output, hidden)

        # Get activations and multiply by temperature coefficient
        if self.opt.ctx == "out":
            dist = self.output_temperature * self.output_proj(
                torch.cat([output.view(-1, output.size(2)), goal], 1))
        else:
            dist = self.output_temperature * self.output_proj(
                output.view(-1, output.size(2)))

        return dist, output.squeeze(0)

    def cuda(self, device_id):
        super(Decoder, self).cuda(device_id)
        self.is_cuda = True
