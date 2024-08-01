# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from lark import Lark

from evariste.envs.mm.env import get_parser, MetamathEnv
from evariste.syntax.deprecated.lexer import MetamathLexer
import os
from pathlib import Path
import logging
import pprint

syntatic = set(["setvar", "wff", "class", "set"])
syntatic_vars = {"wff_var", "class_var", "setvar_var"}

base_grammar = """
start: _TWFF wff -> wff
 | _TCLASS class -> class
 | (_TSETVAR | _TSET) setvar -> setvar

wff: _wff_var -> wff_var
| [wff]

class: _class_var -> class_var
| [class]

setvar: _setvar_var -> setvar_var // Must be a var

_wff_var: [wff_var]

_class_var: [class_var]

_setvar_var: [setvar_var]

%declare [declares]
"""

var_set = {"wff": set(), "setvar": set(), "class": set()}
formula_set = {"wff": [], "setvar": [], "class": []}

# Handle old set.mm databases by mapping set to setvar.
typemap = {"wff": "wff", "set": "setvar", "setvar": "setvar", "class": "class"}

## Lookup table from lark node name to proposition label
label_lookup = {}

## Maps terminal literals to Lark terminal names (all caps)
terminal_lookup = {
    "wff": "_TWFF",
    "class": "_TCLASS",
    "setvar": "_TSETVAR",
    "set": "_TSET",
}

## Maps from the order in the expression to the actual hypothesis order
axiom_hyp_order = {}

ftable = {}


def escape(sym):
    sym_escape = sym.replace("\\", "\\\\")
    sym_escape = sym_escape.replace('"', '\\"')
    return sym_escape


def terminal(sym):
    if sym in terminal_lookup:
        return terminal_lookup[sym]
    else:
        terminal = f"T{len(terminal_lookup)}"
        terminal_lookup[sym] = terminal
        return terminal


def process_wff_axiom(label, assertion):
    hyp = []
    for hlbl, var in assertion["active_f_labels"].items():
        ftype, flbl = var
        var_set[typemap[ftype]].add(flbl)
        ftable[flbl] = (ftype, hlbl)

    lform = []
    axiom_hyp_order[label] = []
    sym_idx = 0
    for sym in assertion["tokens"][1:]:
        is_base = False
        for set_type, vset in var_set.items():
            if sym in vset:
                lform.append(set_type)
                is_base = True
                for idx, hyp in enumerate(assertion["f_hyps"]):
                    if hyp[1] == sym:
                        axiom_hyp_order[label].append(idx)
                sym_idx += 1

        if not is_base:
            lform.append(f"{terminal(sym)}")

    lark_label = f"w{len(label_lookup)}"
    label_lookup[lark_label] = label
    lform_str = " ".join(lform) + " -> " + lark_label + " // " + label
    typ = assertion["tokens"][0]
    formula_set[typ].append(lform_str)  #


if __name__ == "__main__":
    dir = Path(__file__).parent.absolute()
    parser = get_parser()
    args = parser.parse_args()

    set_name = "mm_recent"
    set_path = "resources/metamath/DATASET_2/set.mm"

    # set_name = "recent"
    # set_path = 'resources/metamath/set.mm/set.mm'

    args.database_path = set_path
    args.grammar_objects_file = f"{dir}/grammars/{set_name}_lalr.py"
    args.stop_label = None
    # args.stop_label = 'nfmod'
    # args.stop_label = 'dflim2'
    # args.stop_label = 'dvdsadd2b' #20% in #opsrval is 40%

    args.grammar_file = Path(f"{dir}/grammars/{set_name}_lalr_grammar.lark")
    assert os.path.isfile(args.database_path)

    logging.basicConfig(level=logging.INFO)

    # create Metamath instance
    mm_env = MetamathEnv(
        args.database_path,
        None,
        start_label=args.start_label,
        stop_label=args.stop_label,
        rename_e_hyps=args.rename_e_hyps,
        decompress_proofs=args.decompress_proofs,
        verify_proofs=args.verify_proofs,
        log_level="debug",
    )

    logging.info("Processing proof database ...")
    mm_env.process()

    logging.info("Processing syntatic axioms ...")
    for lbl, (lbl_type, assertion) in mm_env.labels.items():
        if (lbl_type == "$a") and assertion["tokens"][0] in syntatic:
            process_wff_axiom(lbl, assertion)

    logging.info("Forming grammar ...")
    grammar = base_grammar

    for set_type, vset in var_set.items():
        varlist = sorted(vset, key=len, reverse=True)
        lark_vars = " | ".join(["" + terminal(v) + "" for v in varlist])
        grammar = grammar.replace(f"[{set_type}_var]", lark_vars)

    for set_type, fset in formula_set.items():
        lark_formulas = "\n| ".join(fset)
        grammar = grammar.replace(f"[{set_type}]", lark_formulas)

        ### Adding declare for all terminals
        declares = " ".join([t for t in terminal_lookup.values()])
        grammar = grammar.replace("[declares]", declares)

    logging.info(f"Saving grammar to {args.grammar_file} ...")
    with open(str(args.grammar_file), "w") as grammar_file:
        grammar_file.write(grammar)

    logging.info("Checking grammar is valid Lark ...")

    lark = Lark(grammar, parser="lalr", lexer=MetamathLexer)

    logging.info(f"Saving grammar helper objects to {args.grammar_objects_file} ...")

    grammar_objects = (
        "# PREGENERATED BY export_grammar_lalr: helper objects for the parser\n\n"
    )
    grammar_objects += "grammar_file = '" + args.grammar_file.name + "'\n\n"
    grammar_objects += "syntatic = " + pprint.pformat(syntatic) + "\n\n"
    grammar_objects += "syntatic_vars = " + pprint.pformat(syntatic_vars) + "\n\n"
    grammar_objects += "ftable = " + pprint.pformat(ftable) + "\n\n"
    grammar_objects += "label_lookup = " + pprint.pformat(label_lookup) + "\n\n"
    grammar_objects += "axiom_hyp_order = " + pprint.pformat(axiom_hyp_order) + "\n\n"
    grammar_objects += "terminal_lookup = " + pprint.pformat(terminal_lookup) + "\n\n"

    with open(args.grammar_objects_file, "w") as grammar_objects_file:
        grammar_objects_file.write(grammar_objects)

    logging.info("Finished")
