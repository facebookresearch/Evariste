# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle


class Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        remappings = [
            ("generation.replay_buffer", "comms"),
            ("evariste.envs.metamath_utils", "evariste.envs.mm.utils"),
        ]
        for old, new in remappings:
            module = module.replace(old, new)
        if name == "SamplingMethod":
            module = "evariste.refac.deleted_classes"
        if name == "FilterKind":
            module = "evariste.refac.deleted_classes"
        if module == "evariste.forward.common":
            try:
                return super(Unpickler, self).find_class(module, name)
            except AttributeError:
                pass
            module = "evariste.forward.generation_errors"
        return super(Unpickler, self).find_class(module, name)
