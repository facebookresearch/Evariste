# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import sys


def process(filename: str) -> None:
    with open(filename, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("-"):
                data = line.split()
                op = data[0][1]
                if op == "Q":
                    assert len(data) == 3
                    print(f'Add LoadPath "{data[1]}" as {data[2]}.')
                elif op == "R":
                    assert len(data) == 3
                    print(f'Add Rec LoadPath "{data[1]}" as {data[2]}.')
                elif op == "I":
                    assert len(data) == 2
                    print(f'Add ML Path "{data[1]}".')
                else:
                    print(f'## Unsupported operator: "{line}"')
            else:
                print(f'## Skipped: "{line}"'.format(line))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: %s path/to/_CoqProject", sys.argv[0])
        sys.exit(0)
    process(sys.argv[1])
