# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from evariste.refac import pickle as refac_pickle


def safe_load(path, map_location: str):
    """ Allows loading of old models"""
    return torch.load(path, map_location=map_location, pickle_module=refac_pickle)


def safe_pkl_load(file):
    return refac_pickle.Unpickler(file).load()
