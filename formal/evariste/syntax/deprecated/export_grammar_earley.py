# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from lark import Lark

from evariste.envs.mm.env import get_parser, MetamathEnv
import os
from pathlib import Path
import logging
import pprint

syntatic = set(["setvar", "wff", "class", "set"])

base_grammar = """
start: "wff" _WS wff -> wff
 | "class" _WS class -> class
 | ("setvar" | "set") _WS setvar -> setvar

wff: WFF_VAR -> wff_var
| [wff]

class: CLASS_VAR -> class_var
| [class]

setvar: SETVAR_VAR -> setvar_var // Must be a var

WFF_VAR: [wff_var]

CLASS_VAR: [class_var]

SETVAR_VAR: [setvar_var]

_WS: " "
"""

var_set = {"wff": set(), "setvar": set(), "class": set()}
formula_set = {"wff": {}, "setvar": {}, "class": {}}

# Handle old set.mm databases by mapping set to setvar.
typemap = {"wff": "wff", "set": "setvar", "setvar": "setvar", "class": "class"}


## Lookup table from lark node name to proposition label
label_lookup = {}

# Maps from the order in the expression to the actual hypothesis order
axiom_hyp_order = {}

ftable = {}


def escape(sym):
    sym_escape = sym.replace("\\", "\\\\")
    sym_escape = sym_escape.replace('"', '\\"')
    return sym_escape


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
                lform.append("_WS")
                is_base = True
                for idx, hyp in enumerate(assertion["f_hyps"]):
                    if hyp[1] == sym:
                        axiom_hyp_order[label].append(idx)
                sym_idx += 1

        if not is_base:
            lform.append(f'"{escape(sym)}"')
            lform.append("_WS")

    lform = lform[:-1]  # Remove WS at end

    lark_label = f"w{len(label_lookup)}"
    label_lookup[lark_label] = label
    lform_str = " ".join(lform) + " -> " + lark_label + " // " + label
    typ = assertion["tokens"][0]
    formula_set[typ][label] = lform_str


if __name__ == "__main__":
    # parse arguments
    dir = Path(__file__).parent.absolute()
    parser = get_parser()
    args = parser.parse_args()

    # set_name = "holophrasm"
    # set_path = 'resources/metamath/holophrasm/set.mm'

    # set_name = "recent"
    # set_path = "resources/metamath/set.mm/set.mm"

    set_name = "mm_recent"
    set_path = "resources/metamath/DATASET_2/set.mm"

    args.database_path = set_path
    args.grammar_objects_file = f"{dir}/grammars/{set_name}_earley.py"
    args.stop_label = None
    # args.stop_label = 'nfmod'
    # args.stop_label = 'dflim2'
    # args.stop_label = 'dvdsadd2b' #20% in #opsrval is 40%
    args.grammar_file = Path(f"{dir}/grammars/{set_name}_earley_grammar.lark")
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

    logging.info("Calculating usage statistics")
    usages = {}
    for proof in mm_env.decompressed_proofs.values():
        for label in proof:
            if label in usages:
                usages[label] += 1
            else:
                usages[label] = 1

    ### Process axioms to grammar rules
    logging.info("Processing syntatic axioms ...")
    for lbl, (lbl_type, assertion) in mm_env.labels.items():
        if (lbl_type == "$a") and assertion["tokens"][
            0
        ] in syntatic:  # or lbl_type == "$p"
            process_wff_axiom(lbl, assertion)

    logging.info("Forming grammar ...")
    grammar = base_grammar

    ### Variables
    for set_type, vset in var_set.items():

        def sort_key(v):
            ## Sort by length first, and break ties by usage
            x = 100_000 ** len(v)
            x += usages.get(ftable[v][1], 0)
            return x

        varlist = sorted(vset, key=sort_key, reverse=True)
        lark_vars = " | ".join(['"' + escape(v) + '"' for v in varlist])
        grammar = grammar.replace(f"[{set_type}_var]", lark_vars)

    # Formulas
    for set_type, fset in formula_set.items():
        sorted_fset = sorted(
            fset.items(), key=lambda kv: usages.get(kv[0], 0), reverse=True
        )
        lark_formulas = "\n| ".join([formula for label, formula in sorted_fset])
        grammar = grammar.replace(f"[{set_type}]", lark_formulas)

    logging.info("Checking grammar is valid Lark ...")
    lark = Lark(grammar)

    logging.info(f"Saving grammar to {args.grammar_file} ...")
    with open(args.grammar_file, "w") as grammar_file:
        grammar_file.write(grammar)

    logging.info(f"Saving grammar helper objects to {args.grammar_objects_file} ...")

    grammar_objects = (
        "# PREGENERATED BY export_grammar_earley: helper objects for the parser\n\n"
    )
    grammar_objects += "grammar_file = '" + args.grammar_file.name + "'\n\n"
    grammar_objects += "ftable = " + pprint.pformat(ftable) + "\n\n"
    grammar_objects += "label_lookup = " + pprint.pformat(label_lookup) + "\n\n"
    grammar_objects += "axiom_hyp_order = " + pprint.pformat(axiom_hyp_order) + "\n"

    with open(args.grammar_objects_file, "w") as grammar_objects_file:
        grammar_objects_file.write(grammar_objects)

    logging.info("Finished")
