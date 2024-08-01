# distutils: language = c++
from earley cimport Grammar
from libcpp.string cimport string
from lark.tree import Tree
from lark.exceptions import LarkError

def build_tree(input_str, serialized, names):
    if len(serialized) == 0:
        raise LarkError
    start, end = serialized[0], serialized[0] + serialized[1]
    tree = Tree(names[0].decode('utf-8'), [])
    tree.meta.start_pos, tree.meta.end_pos = start, end
    tree.string = input_str[start:end]

    next_ser = 3
    child_name = 1
    for child_id in range(serialized[2]):
        length, child_length, child = build_tree(
            input_str, serialized[next_ser:], names[child_name:]
        )
        child_name += child_length
        tree.children.append(child)
        next_ser += length
    return next_ser, child_name, tree


cdef class PyGrammar:
    cdef Grammar* c_g

    def __cinit__(self, string path):
        self.c_g = new Grammar(path)

    def parse(self, to_parse):
        cdef res = self.c_g.parse(to_parse)
        return build_tree(to_parse, res[0], res[1])[-1]

    def __dealloc__(self):
        del self.c_g