# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
from logging import getLogger


logger = getLogger()


def init_opam_switch(switch_name, check_ocaml=False, check_coq=False):
    """
    Update environment variables to use the given Opam switch.
    """
    assert len(switch_name.strip()) == len(switch_name) > 0
    opam_path = os.path.join(os.getenv("HOME"), ".opam")
    switch_path = os.path.join(opam_path, switch_name)
    assert os.path.isdir(opam_path)
    assert os.path.isdir(switch_path)
    logger.info(f"Updating environment to use {switch_name} switch ({switch_path}) ...")

    # read new variables
    lines = os.popen(f"opam env --switch {switch_name} --set-switch").read()
    lines = [line.strip() for line in lines.split("\n") if len(line.strip()) > 0]

    # update variables
    for line in lines:
        m = re.match(r"([a-zA-Z0-9_]+)='([^ ]*)'; export ([a-zA-Z0-9_]+);", line)
        assert m is not None
        varname, new_content, varname2 = m.groups()
        assert varname == varname2
        logger.info(
            f"Updating variable {varname}\n"
            f"Old: {os.environ.get(varname)}\n"
            f"New: {new_content}"
        )
        os.environ[varname] = new_content

    # check OCaml
    if check_ocaml:
        path = os.popen("which ocaml").read().strip()
        if path == "":
            raise Exception("Unable to locate OCaml")
        version = os.popen("ocaml --version").read().strip()
        assert path == os.path.join(switch_path, "bin", "ocaml"), (
            path,
            os.path.join(switch_path, "bin", "ocaml"),
        )
        logger.info(f"ocaml path: {path}")
        logger.info(f"ocaml version:\n{version}")

    # check Coq
    if check_coq:
        path = os.popen("which coqtop").read().strip()
        if path == "":
            raise Exception("Unable to locate Coq")
        version = os.popen("coqtop --version").read().strip()
        assert path == os.path.join(switch_path, "bin", "coqtop")
        logger.info(f"coqtop path: {path}")
        logger.info(f"coqtop version:\n{version}")
