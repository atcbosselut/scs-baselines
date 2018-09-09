"""
File for methods related to loading configuration files
and seting experiment parameters

author: Antoine Bosselut (atcbosselut)
"""

import json
from jobman import DD

device = 0
do_gpu = False

gpu_mapping = {
    "thor": {
        2: 0,
        3: 1,
        0: 2,
        1: 3
    },
    "default": {
        0: 0,
        1: 1,
        2: 2,
        3: 3
    }
}


def get_parameters(opt, exp="class"):
    params = DD()
    params.net = DD()
    params.net.enc = DD()
    get_encoder_parameters(params, opt)
    params.net.gendec = DD()
    params.net.classdec = DD()

    if exp == "class":
        get_class_parameters(params, opt)
    elif exp == "gen":
        get_gendec_parameters(params, opt)
    else:
        get_class_parameters(params, opt)
        params.net.enc.pt = "none"
    params.train = DD()
    params.train.static = DD()
    params.train.dynamic = DD()
    params.eval = DD()
    params.data = DD()

    # Which experiment to run: {class = classification; gen = generation}
    params.exp = opt.experiment

    # Task to run: {emotion, motivation}
    params.task = opt.task

    if params.exp == "class":
        # % of development set stories to keep as training data
        params.train.static.tr = opt.train_ratio

        # granularity of labels
        # motivation: {maslow, reiss}
        # emotion: {plutchik, plutchik16}
        params.granularity = opt.granularity

        # Labeling type (default = majority)
        params.data.label = opt.label

    if params.exp == "gen":
        # Number of positive examples per negative example
        params.train.static.pnr = opt.pos_neg_ratio

        # Loss to use during training
        params.train.static.wcrit = "nll"

        # Prune useless words in motivation sequences such as
        # "to be" in "to be famous"
        params.data.pruned = opt.pruned

    # Max norm at which to clip gradients
    params.train.static.gc = opt.grad_clip

    # Random seed
    params.train.static.seed = opt.random_seed

    # learning rate
    params.train.dynamic.lr = opt.learning_rate

    # batch size
    params.train.dynamic.bs = opt.batch_size

    # optimizer to use {adam, rmsprop, etc.}
    # Only Adam is actually implemented
    params.train.dynamic.optim = opt.optimizer

    # Default parameters for the CNN model from
    # 2014 Yoon Kim paper
    if params.net.enc.model == "cnn+stock":
        params.net.enc.ks = "3,4,5"
        params.net.enc.kn = 100
        params.net.classdec.dpt = 0.5
        params.train.dynamic.lr = 0.001
        params.train.dynamic.bs = 64
        params.data.shuffle = False
        params.train.static.l2 = 3
        params.net.enc.iSize = 128
        params.net.classdec.hSize = 300

    meta = DD()
    meta.iterations = opt.iterations
    meta.epochs = opt.epochs
    meta.mark = opt.mark

    return params, meta


def get_encoder_parameters(params, opt):
    params.net.enc = DD()

    # Whether to encode the entity context
    # For LSTM and CNN, this adds a separate encoder
    # for entity-specific context lines.
    # For REN and NPN, this does nothing since those models
    # represent entities using memory slots, so the var is
    # overwritten below
    params.net.enc.ctx = opt.encoder.ctx

    # Type of encoder to us -- {lstm, cnn, ren, npn}
    params.net.enc.model = opt.encoder.model

    # Hidden sizes of encoder
    # Self-explanatory for LSTMs
    params.net.enc.hSize = opt.hidden_size

    # Input embedding size
    params.net.enc.iSize = opt.embed_size

    # Dropout probability for any nodes in the encoder
    params.net.enc.dpt = opt.encoder.dropout

    # Type of pretrained embeddings for the encoder
    # Either glove or none (feel free to add other types)
    params.net.enc.pt = opt.encoder.pt

    # Initialization to use
    # d = default xavier glorot initialization
    # Gets overwritten for REN or NPN initialization
    params.net.enc.init = opt.encoder.init

    if params.net.enc.model in ['gru', 'lstm', 'rnn']:
        # num layers in rnn
        params.net.enc.nL = opt.encoder.rnn.num_layers

        # make encoder bidirectional
        params.net.enc.bid = opt.encoder.rnn.bid

    elif params.net.enc.model in ["ren", "npn"]:
        # Set context to be true automatically
        params.net.enc.ctx = True

        # tie entity cells to story entities
        params.net.enc.tied = opt.encoder.ren.tied

        # how many entity slots
        params.net.enc.ents = opt.encoder.ren.num_slots

        # activation function for entity update
        # (P = PReLU or I = Identity)
        params.net.enc.act = opt.encoder.ren.activation

        # Size of entity hidden cells
        params.net.enc.eSize = opt.encoder.ren.entity_size

        # how to intialize parameters
        # format is gauss+{}+{}.format(mean, std)
        # n = the default initialization pytorch
        params.net.enc.init = opt.encoder.ren.init

        # entity_act update function options:
        # k = key projection (REN-style),
        # v = value projection (REN-style),
        # c = context projection (REN-style)}
        params.net.enc.afunc = opt.encoder.ren.application_function

        # use action and affect labels to supervise entitiy selection
        params.net.enc.sup = opt.encoder.ren.supervise

        # lock keys to glove init
        params.net.enc.lk = opt.encoder.ren.lock_keys

        # use glove embeddings for pretrained entities
        params.net.enc.entpt = opt.encoder.ren.entpt

        if params.net.enc.model == "npn":
            # Number of actions
            params.net.enc.na = opt.encoder.npn.actions

            # Size of action embeddings
            params.net.enc.aSize = opt.encoder.npn.action_size

            # Number of MLP layers for selecting actions
            params.net.enc.aNL = opt.encoder.npn.action_num_layers

            # Dropout for action selector MLP
            params.net.enc.adpt = opt.encoder.npn.action_dropout

            # Activation functions between layers of action selector MLP
            params.net.enc.aI = opt.encoder.npn.action_init

            # number of layers of projections for entity selection
            params.net.enc.eNL = opt.encoder.npn.entity_num_layers

            # dropout probability in preprocess layers
            params.net.enc.edpt = opt.encoder.npn.entity_dropout

            # use recurrent attention (See Bosselut et al., 2018)
            params.net.enc.rec = opt.encoder.npn.entity_recurrent_attention

            # Sum entity selection or just scale
            params.net.enc.eRed = opt.encoder.npn.entity_reduce

            # If it's using an NPN, you need to activate action contribution
            # to entity update if you mistakenly haven't done so in
            # params.net.enc.afunc
            if "a" not in params.net.enc.afunc:
                params.net.enc.afunc = "a" + params.net.enc.afunc
        else:
            # EntNet update rule
            params.net.enc.afunc = "nkvc"

            # Initialize encoder with REN initialization
            params.net.enc.init = opt.encoder.ren.init

            # No projection layers between encoder and entity section
            params.net.enc.eNL = 0

    elif params.net.enc.model == "cnn":
        # Size of kernels (different sizes separated by commas)
        params.net.enc.ks = opt.encoder.cnn.kernel_sizes

        # Number of kernel functions
        params.net.enc.kn = opt.encoder.cnn.kernel_num


def get_gendec_parameters(params, opt):
    params.net.gendec = DD()

    # Type of model to use (e.g., ed = simple decoder)
    params.net.gendec.model = opt.generation_decoder.model

    # Recurrent unit to use: {lstm}
    params.net.gendec.unit = opt.generation_decoder.unit

    # NUmber of layers in decoder
    params.net.gendec.nL = opt.generation_decoder.num_layers

    # Dropout
    params.net.gendec.dpt = opt.generation_decoder.dropout

    # Coefficient by which to multiply output activations before softmax
    # Higher makes distribution over tokens more peaky
    # Lower flattens the distribution
    params.net.gendec.dt = opt.generation_decoder.output_temperature

    # Size of hidden state of recurrent unit in decoder
    params.net.gendec.hSize = opt.hidden_size

    # Size of embeddings in the decoder
    params.net.gendec.iSize = opt.embed_size

    # Where to attach context vector
    # (out = concatenate with recurrent unit output)
    # (inp = concatenate with word embedding)
    params.net.gendec.ctx = opt.generation_decoder.context


def get_class_parameters(params, opt):
    params.net.classdec = DD()

    # number of input features to the classifier
    # should be the same as this hidden size of encoder
    params.net.classdec.hSize = opt.hidden_size

    # Dropout in the classifier
    params.net.classdec.dpt = opt.classification_decoder.dropout


def read_config(file_):
    config = DD()
    print file_
    for k, v in file_.iteritems():
        if v == "True" or v == "T" or v == "true":
            config[k] = True
        elif v == "False" or v == "F" or v == "false":
            config[k] = False
        elif type(v) == dict:
            config[k] = read_config(v)
        else:
            config[k] = v

    return config


def load_config(name):
    with open(name, "r") as f:
        config = json.load(f)
    return config


def get_glove_name(opt, type_="tokens", key="pt"):
    emb_type = getattr(opt, key)
    if type_ == "tokens":
        return "data/{}/tokens.{}{}.vocab".format(
            emb_type, emb_type, opt.iSize)
    else:
        return "data/{}/entities.{}{}.vocab".format(
            emb_type, emb_type, opt.eSize)
