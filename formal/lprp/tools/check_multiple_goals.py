# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from matplotlib import pyplot as plt
import numpy as np


def _parse_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target_file")
    parser.add_argument("limit", type=int, default=500000)
    return parser.parse_args()


def _main():
    opts = _parse_main()
    count = 0
    pred = lambda x: x.count("⊢") > 1
    goal2count = dict()
    with open(opts.target_file, "r") as f:
        for l in f:
            if count > opts.limit:
                print("[check_multiple_goals] LIMIT HIT")
                break
            if pred(l):
                count += 1
                print("[check_multiple_goals] FOUND MULTIPLE GOALS: ", l)
            if l in goal2count:
                goal2count[l] = goal2count[l] + 1
            else:
                goal2count[l] = 1

    repeated_goals = [k for k, v in goal2count.items() if v > 1]
    print("REPEATED GOALS", repeated_goals)
    print(
        "REPEATED GOALS WITH MULTIPLE GOALS: ",
        len([x for x in repeated_goals if x.count("⊢") > 1]),
    )
    print("NUM REPEATED GOALS: ", len(repeated_goals))
    print("DUPLICATE CONTRIBUTION: ", sum(goal2count[k] - 1 for k in repeated_goals))
    print("REPETITION DISTRIBUTION")
    bins = np.arange(0, 5, 1)
    plt.xlim([0, 7])
    plt.hist([goal2count[k] for k in repeated_goals])
    plt.show()
    count_repeats = len(repeated_goals)

    print(f"[check_multiple_goals] FOUND {count_repeats} REPEATED GOALS")
    print(f"[check_multiple_goals] FOUND {count} INSTANCES OF MULTIPLE GOALS")


if __name__ == "__main__":
    _main()
