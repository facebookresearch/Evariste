# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from evariste import json as json
from evariste.envs.mm.env import MetamathEnv

syntactic = {"setvar", "wff", "class", "set"}

base_grammar = """$start wff $wff
$start class $class
$start set $setvar
$start setvar $setvar"""

var_set = {"wff": set(), "setvar": set(), "class": set()}
formula_set = {"wff": {}, "setvar": {}, "class": {}}

# Handle old set.mm databases by mapping set to setvar.
typemap = {"wff": "wff", "set": "setvar", "setvar": "setvar", "class": "class"}

# c2 cadd c2 co => ( 2 2 + ) we need to reorder things when building the final proof from the parse tree
axiom_hyp_order = {}

# ph -> wph
xx_var_to_axiom = {}

label_lookup = {}

ftable = {}


def process_wff_axiom(label, assertion):
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
                lform.append(f"${set_type}")
                is_base = True
                sym_idx += 1
                for idx, hyp in enumerate(assertion["f_hyps"]):
                    if hyp[1] == sym:
                        axiom_hyp_order[label].append(idx)

        if not is_base:
            lform.append(f"{sym}")

    lform_str = " ".join(lform) + f" $*{label}"
    typ = assertion["tokens"][0]
    formula_set[typ][label] = lform_str


if __name__ == "__main__":

    set_path = ""

    database_path = set_path
    logging.basicConfig(level=logging.INFO)

    # create Metamath instance
    mm_env = MetamathEnv(
        database_path,
        None,
        decompress_proofs=False,
        verify_proofs=False,
        log_level="debug",
    )

    logging.info("Processing proof database ...")
    mm_env.process()

    ### Process axioms to grammar rules
    logging.info("Processing syntactic axioms ...")
    for lbl, (lbl_type, assertion) in mm_env.labels.items():
        if lbl_type != "$a":
            continue
        if assertion.tokens[0] in syntactic:
            process_wff_axiom(lbl, assertion)

    logging.info("Forming grammar ...")
    grammar = base_grammar

    ### Variables
    for set_type, vset in var_set.items():
        varlist = sorted(vset, reverse=True)
        grammar = "\n".join([f"#{set_type}# {' '.join(varlist)}"]) + "\n" + grammar

    # Formulas
    for set_type, fset in formula_set.items():
        sorted_fset = sorted(fset.items(), reverse=True)
        for label, formula in sorted_fset:
            grammar += f"\n${set_type} {formula}"
        grammar += f"\n${set_type} #{set_type}# $*{set_type}_var"

    with open("grammar.in", "w") as grammar_out:
        grammar_out.write(grammar)
        grammar_out.write("\n" + "axiom_hyp_order " + json.dumps(axiom_hyp_order))
        grammar_out.write("\n" + "f_table " + json.dumps(ftable) + "\n")

    logging.info("Finished")
