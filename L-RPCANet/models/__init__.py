from .LRPCANet import *


def get_model(name, net=None):
    if name == 'lrpcanet':
        net = LRPCANet(stage_num=6)
    else:
        raise NotImplementedError

    return net
