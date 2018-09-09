"""
File for defining the NPN class and related methods.

author: Antoine Bosselut (atcbosselut)
"""

import src.data.config as cfg
import src.models.models as models

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
import pickle


class EntityUpdater(nn.Module):
    """
    Update entities using entity values changed by applicator

    Initialization Args:
        opt.eSize: size of entities
        opt.afunc: function to use for entity update
                    a = include applicator output (probably should always)
                    k = include projected keys (from REN paper)
                    v = include projected values (from REN paper)
                    c = include projected context (from REN paper)

    Input:
        entities: entity vals at start of NPN cycle
                  (batch_size x num_entities x entity_size)
        new_entities: new_entity from applicator
                  (batch_size x entity_size)
        dist: attention distribution from selecting entities for change
        keys: key vector for entities
        ctx (default=None): context from sentence encoder

    Output:
        updated entity vectors (batch_size x num_entities x entity_size)

    """
    def __init__(self, opt):
        super(EntityUpdater, self).__init__()
        self.opt = opt

        # Update entities with projected context (See EntNet Paper)
        if "c" in self.opt.afunc:
            print "Context Application"
            self.ctx_applicator = nn.Linear(
                opt.hSize, opt.eSize, False)

        self.act = nn.PReLU(opt.eSize)

        # Initialize PReLU negative zone slope to 1
        if "1" in self.opt.act:
            self.act.weight.data.fill_(1)

    def forward(self, entities, new_entities, dist, keys, ctx=None):
        batch_size = entities.size(0)
        num_items = entities.size(1)
        hidden_size = entities.size(2)

        # get update values from context and from applicator
        oc, oa = self.compute_update_components(
            entities, new_entities, keys, ctx)

        # Format attention weights for memory overwrite
        n_dist = dist.view(batch_size, num_items, 1).expand(
            batch_size, num_items, hidden_size)

        # Update the entity memory
        new_ents = self.update_entities(oc, oa, n_dist, entities)

        # Normalize entities
        return new_ents * (1 / new_ents.norm(2, 2)).unsqueeze(2).expand(
            batch_size, num_items, hidden_size), n_dist

    def update_entities(self, oc, oa, n_dist, entities):
        batch_size = entities.size(0)
        num_items = entities.size(1)
        hidden_size = entities.size(2)

        # Just add on contribution of new entities to old ones
        if "n" not in self.opt.afunc:
            # Identity activation
            if self.opt.act == "I":
                new_ents = ((n_dist * (oa + oc)) + (1 - n_dist) *
                            entities)
            # PReLU activation
            else:
                pre_act = (oa + oc).view(-1, hidden_size)
                new_ents = ((n_dist * self.act(pre_act).view(
                    batch_size, num_items, -1)) + (1 - n_dist) * entities)
        # Do interplation of previous entities and current entities
        else:
            if self.opt.act == "I":
                new_ents = entities + n_dist * (oa + oc)
            else:
                pre_act = (oa + oc).view(-1, hidden_size)
                new_ents = (n_dist * self.act(pre_act).view(
                    batch_size, num_items, -1) + entities)

        return new_ents

    def compute_update_components(self, entities, new_entities, keys, ctx):
        batch_size = entities.size(0)
        num_items = entities.size(1)
        hidden_size = entities.size(2)

        oc = 0  # context contribution
        oa = 0  # NPN-changed entity contribtuion

        # NPN contribution
        if new_entities.dim() == 3:
            oa = new_entities
        else:
            oa = new_entities.view(batch_size, 1, hidden_size).expand(
                batch_size, num_items, hidden_size)

        # Context contribution from REN paper
        oc = self.ctx_applicator(ctx).view(batch_size, 1, -1).repeat(
            1, num_items, 1)

        return oc, oa

    def cuda(self, device_id):
        super(EntityUpdater, self).cuda(device_id)
        self.is_cuda = True
        self.act.cuda(device_id)


class EntitySelector(nn.Module):
    """
    Select entities using hidden state from encoder

    Input:
        hidden: hidden state from encoder
        entities: entity vals at start of EntNet cycle
                  (batch_size x num_entities x entity_size)
        keys: key vector for entities
        prev_attn: previous step's attention distribution

    Output:
        selected_entities: selected entity vectors from attention distribution
        entity_dist: attention distribution over entities
        acts: activations of attention distribution
    """
    def __init__(self, opt):
        super(EntitySelector, self).__init__()

        self.initialize_attention(opt)

        # Initialize ppreprocessing projections of
        # sentence encoder hiddens state
        self.preprocess = nn.Sequential()
        for i in range(opt.eNL):
            self.preprocess.add_module(
                "proj_{}".format(i), nn.Linear(opt.hSize, opt.hSize))
            self.preprocess.add_module(
                "act_{}".format(i), nn.PReLU(opt.hSize))
            if opt.edpt != 0 and i < opt.eNL - 1:
                self.preprocess.add_module(
                    "edpt_{}".format(i), nn.Dropout(opt.edpt))

        self.opt = opt

    def initialize_attention(self, opt):
        # Initialize attention function
        # Use bilinear attention
        self.attention = models.BilinearAttention(
            2 * opt.eSize, opt.hSize, 1,
            bias=False, act="sigmoid")

        # If recurrent attention is used,
        # initialize it
        if opt.rec:
            self.choice = nn.Sequential(nn.Linear(opt.hSize, 2),
                                        nn.Softmax(dim=1))
            self.choice_attender = models.Attender()

        self.attender = models.Attender()

    def forward(self, hidden, entities, keys, prev_attn):
        # Transform sentence encoder representation
        preprocessed = self.preprocess(hidden)

        # Compute attention with respect to entity memory
        dist, acts = self.forward_attention(keys, preprocessed, entities)

        # Use recurrent attention
        entity_dist, choice_dist = self.recurrent_attention(
            dist, preprocessed, prev_attn)

        # Attend to entities with final distribution
        selected_entities = self.attender(
            entities, entity_dist, self.opt.eRed)

        return selected_entities, entity_dist, acts

    def forward_attention(self, keys, preprocessed, entities):
        # Attend preprocessed hidden state to entities
        dist, act = self.attention(torch.cat(
            [entities, keys], 2), preprocessed)
        return dist, act

    def recurrent_attention(self, dist, preprocessed, prev_attn):
        batch_size = preprocessed.size(0)
        if self.opt.rec and prev_attn is not None:
            # Compute choice over whether to use current step's attention
            # or fall back to previous step's entity attention
            choice_dist = self.choice(preprocessed)
            return self.choice_attender(torch.cat(
                [prev_attn.view(batch_size, 1, -1),
                 dist.view(batch_size, 1, -1)], 1), choice_dist), choice_dist
        else:
            return dist, None

    def cuda(self, device_id):
        super(EntitySelector, self).cuda(device_id)
        self.attention.cuda(device_id)
        if self.opt.rec:
            self.choice.cuda(device_id)
            self.choice_attender.cuda(device_id)
        self.attender.cuda(device_id)
        self.preprocess.cuda(device_id)
        self.is_cuda = True


class BilinearApplicator(nn.Module):
    """
    Transforms selected entity embeddings by action functions

    Initialization Args:
        opt.aSize: size of action function embeddings
        opt.eSize: size of entity embeddings

    Input:
        actions: weighted sum of action functions
        entities: entity embeddings or all entity embeddings

    Output:
        output: encoded sentence embedding

    """
    def __init__(self, opt):
        super(BilinearApplicator, self).__init__()
        self.applicator = nn.Bilinear(opt.aSize, opt.eSize,
                                      opt.eSize, False)

    def forward(self, actions, entities):
        # If entity embeddings haven't been reduced by their attention
        # Apply action function directly to each entity embedding
        if entities.dim() == 3:
            bs = entities.size(0)
            num_ents = entities.size(1)
            h_dim = entities.size(2)
            a_dim = actions.size(1)

            parallel_entities = entities.view(-1, h_dim)
            repeated_actions = actions.view(bs, 1, a_dim).repeat(
                1, num_ents, 1).contiguous().view(-1, a_dim)

            out = self.applicator(repeated_actions, parallel_entities)

            return out.view(entities.size(0),
                            entities.size(1),
                            entities.size(2))
        # apply average action function to average entity embedding
        else:
            return self.applicator(actions, entities)

    def cuda(self, device_id):
        super(BilinearApplicator, self).cuda(device_id)
        self.applicator.cuda(device_id)
        self.is_cuda = True


class SentenceEncoder(nn.Module):
    """
    Wrapper class for a sentence encoder

    Initialization Args:
        opt: options to pass to sentence encoder class
        max_words: maximum number of words in any sentence in
                   data

    Input:
        input: word indices for a sentence (batch_size x num_words)
        lengths: number of non-pad indices in each sentence in the batch

    Output:
        output: encoded sentence embedding

    """
    def __init__(self, opt, max_words=None):
        super(SentenceEncoder, self).__init__()
        self.opt = opt
        self.sentence_encoder = models.WeightedBOW(opt, max_words)

    def forward(self, input, lengths):
        output, hidden = self.sentence_encoder(
            input, lengths)

        return output

    def cuda(self, device_id):
        super(SentenceEncoder, self).cuda(device_id)
        self.sentence_encoder.cuda(device_id)
        self.is_cuda = True


class ActionSelector(nn.Module):
    """
    Compute attention distribution over actions
    Compute average action embedding from this attention distribution

    Initialization Args:
        opt.na: # of actions to initialize
        opt.aSize: size of action embeddings to initialize
        opt.adpt: dropout between MLP layers
        opt.hSize: hidden size of MLP layers
        opt.aNL: number of MLP layers


    Input:
        input: encoded sentence (batch_size x opt.hSize)

    Output:
        average action from selecting over action set

    """
    def __init__(self, opt):
        super(ActionSelector, self).__init__()

        self.selector = nn.Sequential()

        # initialize actions
        self.actions = nn.Parameter(torch.FloatTensor(
            opt.na, opt.aSize))
        stdv = 1. / math.sqrt(self.actions.size(1))
        self.actions.data.uniform_(-stdv, stdv)

        # Build MLP for computing attention distribution over actions
        for i in range(opt.aNL):

            # Add fully connected layer
            self.selector.add_module(
                "proj_{}".format(i), nn.Linear(opt.hSize, opt.hSize))

            # Add activation
            self.selector.add_module(
                "act_{}".format(i), nn.Tanh())

            if opt.adpt != 0:
                # Add dropout layer
                self.selector.add_module(
                    "adpt_{}".format(i), nn.Dropout(opt.adpt))

        # Projection vocabulary
        self.selector.add_module(
            "proj_final", nn.Linear(opt.hSize, opt.na))

        # In actual paper we use a Sigmoid, but since these actions
        # are implicit, I just made it a softmax
        # If we actually ground actions to concepts, we can convert this
        # to a sigmoid and have an L1 normalization step after for the
        # attention distributions
        self.selector.add_module(
            "choose", nn.Softmax(dim=1))

    def forward(self, input):
        return torch.mm(self.selector(input), self.actions)

    def cuda(self, device_id):
        super(ActionSelector, self).cuda(device_id)
        self.selector.cuda(device_id)
        self.actions.cuda(device_id)
        self.is_cuda = True


class NPN(nn.Module):
    """
    Neural Process Network wrapper class that wraps around the
    individual components of the network

    Initialization Args:
        max_words: maximum number of words in input sentences
                   (useful for positional mask in REN paper)
        q_max_words: maximum number of words in questions
                    (useful for positional mask in REN paper)
        opt: options for all model components
        opt.ent: number of entities to initialize
        opt.eSize: size of entity embeddings to initialize

    Input:
        sentence: list of two components
        sentence[0]: Tensor of sentences (batch_size x seq_len)
                    -- might be other way around
        sentence[1]: tensor of sentences lengths (batch_size) for
                     sentences in batch
        question: same as for sentence
        answers: tensor of answer values
        prev_attn: attention for selecting entities from previous step
                   of updating entities. Recurrent Attention in paper

    Output:
        If sentence:
        new entity values
        combined output
        selection loss if relevant



    """
    def __init__(self, opt, max_words=None):
        super(NPN, self).__init__()
        # Initialize encoder for entity selection
        self.encoder = SentenceEncoder(opt, max_words)

        # Initialize separate action encoder
        self.action_enc = SentenceEncoder(opt, max_words)

        # Initialize action selector
        self.action_selector = ActionSelector(opt)

        # Initialize applicator for agents and themes
        self.theme_applicator = BilinearApplicator(opt)

        # Initialize entity selector
        self.entity_selector = EntitySelector(opt)

        # Initialize entity updater
        self.entity_updater = EntityUpdater(opt)

        # Initialize entity embeddings
        self.key_init = nn.Embedding(opt.ents, opt.eSize)

        self.crit = nn.BCEWithLogitsLoss()

        self.opt = opt

        self.is_cuda = False

    def initialize_entities(self, entity_ids=None, batch_size=32):
        if entity_ids is None:
            entity_ids = Variable(torch.LongTensor(range(
                self.opt.ents)).view(
                1, self.opt.ents).expand(
                batch_size, self.opt.ents))
            if self.is_cuda:
                entity_ids = entity_ids.cuda(cfg.device)
        keys = self.key_init(entity_ids)

        if self.opt.lk:
            keys = keys.detach()

        self.keys = keys

        entities = self.key_init(entity_ids.detach())
        return self.keys, entities, None

    def forward(self, sentences, sentence_lengths,
                entity_labels=None, entity_init=None):
        bs = sentences[sentences.keys()[0]].size(0)
        keys, entities, _ = self.initialize_entities(
            entity_ids=entity_init, batch_size=bs)

        sel_loss = 0

        attn_dist = None

        for i in sorted(sentences.keys()):
            sentence = sentences[i]

            # Encode sentence, sentence[0] = sentence tensors
            # sentence[1] = sentence length tensors (need this for padding)
            sent_emb = self.encoder(
                Variable(sentence), sentence_lengths[i])

            act_emb = self.action_enc(
                Variable(sentence), sentence_lengths[i])

            # Select actions
            actions = self.action_selector(act_emb)

            # Select entities
            selected_entities, attn_dist, attn_acts = self.entity_selector(
                sent_emb, entities, self.keys, attn_dist)

            if entity_labels is not None:  # and entity_labels[i].sum() != 0:
                sel_loss += self.crit(attn_acts, Variable(entity_labels[i]))

            # Apply action to entities
            changed_entities = self.theme_applicator(
                actions, selected_entities)

            # Upate entities
            entities, n_dist = self.entity_updater(
                entities, changed_entities, attn_dist, self.keys, sent_emb)

            joint = (n_dist * entities).sum(1)

        return entities, joint, sel_loss

    def load_entity_embeddings(self, vocab):
        if self.opt.pt != "none":
            name = "data/{}/entities.{}{}.vocab".format(
                self.opt.pt, self.opt.entpt,
                self.opt.eSize)

            print "Loading entity embeddings from {}".format(name)
            with open(name, "r") as f:
                entity_words = pickle.load(f)

            for i, word in vocab.iteritems():
                if word in ["<unk>", "<start>", "<end>", "<pad>"]:
                    self.key_init.weight.data[i].zero_()
                    continue
                if self.is_cuda:
                    vec = torch.cuda.FloatTensor(entity_words[word])
                else:
                    vec = torch.FloatTensor(entity_words[word])
                self.key_init.weight.data[i] = vec

    def initialize_actions(self, init, vocab):
        if init == "n":
            # Xavier initialization
            stdv = 1. / math.sqrt(self.action_selector.actions.size(1))
            self.action_selector.actions.data.uniform_(-stdv, stdv)
        elif "gauss" in init:
            # Gaussian initialization
            deets = init.split("+")
            mean = deets[1]
            stddev = deets[2]
            print("Using Gaussian Initialization with Mean = {} and \
                Standard Deviation = {} for actions").format(
                float(mean), float(stddev))
            self.action_selector.actions.data.normal_(
                float(mean), float(stddev))
        else:
            raise

    def cuda(self, device_id):
        super(NPN, self).cuda(device_id)
        self.encoder.cuda(device_id)
        self.action_selector.cuda(device_id)
        self.theme_applicator.cuda(device_id)
        self.entity_selector.cuda(device_id)
        self.entity_updater.cuda(device_id)
        self.key_init.cuda(device_id)

        self.action_enc.cuda(device_id)
        # self.entity_init.cuda(device_id)
        self.crit.cuda(device_id)
        self.is_cuda = True
