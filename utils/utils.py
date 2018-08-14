from distutils.dir_util import mkpath


# is_dir look ast at whether the name we make
# should be a directory or a filename
def make_name(opt, prefix="", eval_=False, is_dir=True, set_epoch=None):
    string = prefix
    string += "{}_{}".format(opt.exp, opt.task)
    if opt.exp == "class":
        string += "_{}".format(opt.granularity)
    string += "/"
    string += make_name_string(opt.data) + "/"
    for k, v in opt.net.iteritems():
        string += "{}={}/".format(k, make_name_string(v))
    string += make_name_string(opt.train.static) + "/"
    if eval_:
        string += make_name_string(opt.eval) + "/"
    # mkpath caches whether a directory has been created
    # In IPython, this can be a problem if the kernel is
    # not reset after a dir is deleted. Trying to recreate
    # that dir will be a problem because mkpath will think
    # the directory already exists
    if not is_dir:
        mkpath(string)
    string += make_name_string(opt.train.dynamic, True, set_epoch)
    if is_dir:
        mkpath(string)

    return string


# Make file name portion for this part of the parameter dictionary
def make_name_string(dict_, final=False, set_epoch=None):
    if final:
        if set_epoch is not None:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, set_epoch)
        else:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, dict_.epoch)

        return string

    string = ""

    for k, v in dict_.iteritems():
        if str(v) == "False":
            val = "F"
        elif str(v) == "True":
            val = "T"
        else:
            val = v
        if string:
            string += "-"
        string += "{}_{}".format(k, val)

    return string
