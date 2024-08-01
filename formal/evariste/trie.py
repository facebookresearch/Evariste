# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple


class TrieNode:
    def __init__(self, token):
        self.token = token
        self.children = {}
        self.terminal = False
        self.counter = 1

    def __str__(self):
        s = []
        for k, v in sorted(self.children.items()):
            s.append(f"{k} ({v.counter})")
            if len(v.children) > 0:
                s.append("  " + str(v).replace("\n", "\n  "))
        return "\n".join(s)


class Trie:
    """
    Trie node implementation.
    Operates on token lists (and not strings).
    """

    def __init__(self):
        self.root = TrieNode(None)

    def add(self, tokens: List[str]):
        """
        Adding a list in the trie structure.
        """
        assert type(tokens) is list
        if self.contains(tokens)[0]:
            return
        node = self.root
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                node.counter += 1
            else:
                new_node = TrieNode(token)
                node.children[token] = new_node
                node = new_node
        node.terminal = True

    def contains(self, tokens: List[str]) -> Tuple[bool, int]:
        """
        Check and return:
          1. Whether the input token list belongs to the trie.
          2. How many times it has been seen as a prefix.
        """
        assert type(tokens) is list
        node = self.root
        for token in tokens:
            if token in node.children:
                node = node.children[token]
            else:
                return False, 0
        return node.terminal, node.counter

    def find_prefixes(self, tokens: List[str]) -> List[int]:
        """
        Retrieve the list of terminal sublists prefix to the input.
        """
        node = self.root
        lengths = []
        for i, token in enumerate(tokens):
            if token not in node.children:
                break
            node = node.children[token]
            if node.terminal:
                lengths.append(i + 1)
        return lengths

    def __len__(self):
        """
        Return the number of inserted elements in the trie,
        i.e. the number of terminal nodes.
        """
        count = 0
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            count += int(node.terminal)
            for c in node.children.values():
                stack.append(c)
        return count

    def __str__(self):
        return str(self.root)


if __name__ == "__main__":

    trie = Trie()

    # populate trie
    for x in [
        "c1 cA ccos cfv",
        "c1 cA ccos cfv",
        "c1 cA ccos cfv cB ccos",
        "c1 cA ccos cfv cA csin cfv cdiv",
        "c1 cA ccos bla bla bli",
        "c1 cA ccos cfv cA csin",
    ]:
        print(f"Adding: {x}")
        trie.add(x.split())

    # print trie
    print("")
    print(trie)
    print(f"Number of elements: {len(trie)}")

    # test contains
    print("")
    for x in [
        "c1 cA",
        "c1 cA ccos",
        "c1 cA ccos cfv",
        "c1 cA ccos cfv cA csin cfv cdiv",
        "c2 bla",
    ]:
        print(f"Contains: {x}")
        print("  " + str(trie.contains(x.split())))

    # test prefixes
    print("")
    for x in [
        "c1 cA",
        "c1 cA ccos",
        "c1 cA ccos cfv",
        "c1 cA ccos cfv cA csin cfv cdiv",
        "c2 bla",
    ]:
        print(f"Prefixes: {x}")
        lengths = trie.find_prefixes(x.split())
        for i in lengths:
            print("          " + " ".join(x.split()[:i]))
