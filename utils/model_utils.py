import torch.nn as nn
import torch.nn.functional as F


def modify_output_for_crit(crit, output):
    if crit == "ce":
        return output
    if crit == "mse":
        return F.softmax(output, dim=1)
    if crit == "nll":
        return F.log_softmax(output, dim=1)


def crits(type_, eval_mode=False, **kwargs):
    # print type_
    # print eval_mode
    if type_ == "mse":
        return nn.MSELoss(**kwargs)
    elif type_ == "ce":
        return nn.CrossEntropyLoss(**kwargs)
    else:
        return nn.NLLLoss(**kwargs)
