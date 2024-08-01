# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

class Average:
    def __init__(self, name=None):
        self.name = name
        self.n = 0
        self.avg = float(0)

    def update(self, x):
        self.avg = float(self.n / (self.n + 1)) * self.avg + float(x / (self.n + 1))
        self.n += 1

    def __repr__(self):
        return (self.name if self.name is not None else "") + " " + str(self.avg)


def count_matching_braces(inp: str, left: str = "{", right: str = "}"):
    left_count = 0
    stack = []
    max_depth = 0
    for c in inp:
        if c == left:
            stack.append(left)
            max_depth = max(max_depth, len(stack))
            left_count += 1
        elif c == right:
            stack.pop(-1)
        else:
            pass
    return left_count, max_depth
