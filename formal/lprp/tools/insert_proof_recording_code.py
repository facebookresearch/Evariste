# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import List, Optional, Tuple

from lean_modifier import LeanModifier
from tokenize_lean_files import LeanFile, TokenType
from parse_lean import AST, LeanParser

LEAN_DIRS = [Path("_target/deps/lean/library/"), Path("_target/deps/mathlib/src/")]

TACTIC_LEAN_FILE = Path("_target/deps/lean/library/init/meta/tactic.lean")


def get_tactic_lean_file_modifications_prefix(sexp: bool):
    x = "sexp" if sexp else "flat"
    return Path(f"lean_modifications/print_tactic_state_{x}.lean")


TACTIC_LEAN_FILE_MODIFICATIONS = Path("lean_modifications/tactic_modifications.lean")

INTERACTIVE_BASE_LEAN_FILE = Path(
    "_target/deps/lean/library/init/meta/interactive_base.lean"
)
INTERACTIVE_BASE_LEAN_FILE_MODIFICATIONS = Path(
    "lean_modifications/interactive_base_modifications.lean"
)

TACTIC_INTERACTIVE_LEAN_FILE = Path(
    "_target/deps/lean/library/init/meta/interactive.lean"
)
TACTIC_INTERACTIVE_LEAN_FILE_MODIFICATIONS = Path(
    "lean_modifications/tactic_interactive_modifications.lean"
)

CONV_INTERACTIVE_LEAN_FILE = Path(
    "_target/deps/lean/library/init/meta/converter/interactive.lean"
)
CONV_INTERACTIVE_LEAN_FILE_MODIFICATIONS = Path(
    "lean_modifications/conv_interactive_modifications.lean"
)

ISTEP_CODE = """meta def istep {α : Type u} (line0 col0 : ℕ) (line col : ℕ) (t : tactic α) : tactic unit :=
λ s, (@scope_trace _ line col (λ _, step t s)).clamp_pos line0 line col
"""

TACTIC_ITACTIC_CODE = """meta def itactic : Type :=
tactic unit
"""

CONV_ITACTIC_CODE = """meta def itactic : Type :=
conv unit
"""

DO_NOT_MODIFY = []


def find_code_location(file: Path, code_string: str) -> Tuple[int, int]:
    code_lines = [l.strip() for l in code_string.strip().split("\n")]
    with open(file, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if all(
                lines[i + j].strip() == code_lines[j] for j in range(len(code_lines))
            ):
                return (i, i + len(code_lines))
    raise Exception(
        f"Cannot find an exact instance of the below code in the file {file}:\n{code_string}"
    )


def get_modification(modification_lean: Path) -> str:
    with open(modification_lean, "r") as f:
        lines = []
        record = False
        for l in f:
            if l.startswith("--PR END MODIFICATION"):
                return "".join(lines)
            elif record:
                lines.append(l)
            elif l.startswith("--PR BEGIN MODIFICATION"):
                record = True
    raise Exception("No modification found.")


def insert_tactic_tracing_code(dryrun: bool, sexp: bool):
    l1, l2 = find_code_location(TACTIC_LEAN_FILE, ISTEP_CODE)
    tactic_recording_code_prefix = get_modification(
        get_tactic_lean_file_modifications_prefix(sexp)
    )
    tactic_recording_code = get_modification(TACTIC_LEAN_FILE_MODIFICATIONS)

    prefix_modifier = LeanModifier(TACTIC_LEAN_FILE)
    prefix_modifier.delete_lines(l1, l2)
    prefix_modifier.add_lines_at_end(tactic_recording_code_prefix)
    prefix_modifier.build_file(dryrun=dryrun)

    modifier = LeanModifier(TACTIC_LEAN_FILE)
    # modifier.delete_lines(l1, l2)
    modifier.add_lines_at_end(tactic_recording_code)
    modifier.build_file(dryrun=dryrun)


def insert_param_tracing_code(dryrun: bool):
    # interactive base
    tactic_recording_code = get_modification(INTERACTIVE_BASE_LEAN_FILE_MODIFICATIONS)
    modifier = LeanModifier(INTERACTIVE_BASE_LEAN_FILE)
    modifier.add_lines_at_end(tactic_recording_code)
    modifier.build_file(dryrun=dryrun)

    # (tactic) interactive
    _, l2 = find_code_location(TACTIC_INTERACTIVE_LEAN_FILE, TACTIC_ITACTIC_CODE)
    tactic_recording_code = get_modification(TACTIC_INTERACTIVE_LEAN_FILE_MODIFICATIONS)
    modifier = LeanModifier(TACTIC_INTERACTIVE_LEAN_FILE)
    modifier.add_lines(l2, tactic_recording_code)
    modifier.build_file(dryrun=dryrun)


@dataclass
class Position:
    line: int
    column: int
    end_line: int
    end_column: int


@dataclass
class Modification:
    line: int
    end_line: int
    new_lines: str


@dataclass
class InteractiveParameter:
    pos: Position
    command: str
    param_ix: int
    parser_pos: Optional[Position] = None


class ModifyInterativeParameters:
    """
    Find all parameters of the form `(p : parse <parser>)` and `(t : itactic)`.
    Modify them to have proof tracing code.
    """

    @staticmethod
    def expr_begins_with(expr: AST.Expr, tokens: List[str]) -> bool:
        if len(expr.expr_parts) < len(tokens):
            return False
        for e, t in zip(expr.expr_parts, tokens):
            if not isinstance(e, AST.TokenExprPart):
                return False
            elif e.t_string != t:
                return False
        return True

    @staticmethod
    def extract_parser(
        param_expr: AST.Expr, param_ix: int, start: int
    ) -> InteractiveParameter:
        # strip leading whitepace and $
        parts = param_expr.expr_parts
        for i in range(start, len(parts)):
            part = parts[i]
            if isinstance(part, AST.TokenExprPart):
                if part.t_string == "$":
                    continue
                if part.t_string.strip() == "":
                    continue

            return InteractiveParameter(
                pos=Position(
                    parts[0].line,
                    parts[0].column,
                    parts[-1].end_line,
                    parts[-1].end_column,
                ),
                command="parse",
                param_ix=param_ix,
                parser_pos=Position(
                    parts[i].line,
                    parts[i].column,
                    parts[-1].end_line,
                    parts[-1].end_column,
                ),
            )
        raise Exception("Could not extract parser.")

    def process_param_expr(
        self, param_expr: AST.Expr, param_ix: int
    ) -> Optional[InteractiveParameter]:
        for parse_command in [["parse"], ["interactive", ".", "parse"]]:
            if self.expr_begins_with(param_expr, parse_command):
                return self.extract_parser(param_expr, param_ix, len(parse_command))
        for itactic_command in [
            ["itactic"],
            ["interactive", ".", "itactic"],
            ["tactic", ".", "interactive", ".", "itactic"],
            ["conv", ".", "interactive", ".", "itactic"],
        ]:
            if self.expr_begins_with(param_expr, itactic_command):
                return InteractiveParameter(
                    pos=Position(
                        param_expr.expr_parts[0].line,
                        param_expr.expr_parts[0].column,
                        param_expr.expr_parts[len(itactic_command) - 1].end_line,
                        param_expr.expr_parts[len(itactic_command) - 1].end_column,
                    ),
                    command="".join(itactic_command),
                    param_ix=param_ix,
                )
        return None

    def interactive_parameters(self, ast: AST.DefLike) -> List[InteractiveParameter]:
        interactive_params = []
        param_ix = -1
        for param in ast.signature.params:
            param_ix += 1
            result = self.process_param_expr(param.type_expr, param_ix)
            if result is not None:
                interactive_params.append(result)
        if ast.signature.signature_type is not None:
            for param in ast.signature.signature_type.arg_types:
                param_ix += 1
                result = self.process_param_expr(param, param_ix)
                if result is not None:
                    interactive_params.append(result)
        return interactive_params

    def modify_def(
        self,
        lean_file: LeanFile,
        line: int,
        end_line: int,
        tactic_name: str,
        monad_type: str,
        iparams: List[InteractiveParameter],
    ) -> Modification:
        # reconstruct new declaration
        s = ""
        pos = (line, 0)
        for param in iparams:
            s += lean_file.slice_string(
                pos[0], pos[1], param.pos.line, param.pos.column
            )
            if param.command.endswith("itactic"):
                if param.command.startswith("tactic"):
                    parser = "tactic.interactive.pr.recorded_itactic"
                elif param.command.startswith("conv"):
                    parser = "conv.interactive.pr.recorded_itactic"
                elif monad_type == "tactic":
                    # use the monad type to determine which itactic to use
                    parser = "tactic.interactive.pr.recorded_itactic"
                elif monad_type == "conv":
                    # use the monad type to determine which itactic to use
                    parser = "conv.interactive.pr.recorded_itactic"
                else:
                    raise Exception(f"Unexpected monad type: {monad_type}")

                # Conv not working right, so skip it
                if parser.startswith("tactic"):
                    s += "interactive.parse ("
                    s += parser
                    # s += " " + json.dumps(tactic_name)  # easiest way to escape strings
                    # s += " " + str(param.param_ix)
                    s += ") "
                else:
                    s += param.command
            elif param.command == "parse":
                s += "interactive.parse (interactive.pr.recorded"
                # s += " " + json.dumps(tactic_name)  # easiest way to escape strings
                # s += " " + str(param.param_ix)
                parser = lean_file.slice_string(
                    param.parser_pos.line,
                    param.parser_pos.column,
                    param.parser_pos.end_line,
                    param.parser_pos.end_column,
                )
                parser = parser.strip()
                if parser.startswith("(") and parser.endswith(")"):
                    parser = parser[1:-1].strip()
                # s += " " + json.dumps(parser)  # easiest way to escape strings
                s += " (" + parser + ")) "
            else:
                assert False
            pos = (param.pos.end_line, param.pos.end_column)
        s += lean_file.slice_string(
            pos[0], pos[1], end_line, 0
        )  # Go to end of the line (including newline)

        return Modification(line=line, end_line=end_line, new_lines=s)

    def monad_type(self, lean_file: LeanFile, ast: AST.DefLike) -> Optional[str]:
        if ast.signature.signature_type is not None:
            result_type = ast.signature.signature_type.result_type
            type_string = lean_file.slice_string(
                result_type.line,
                result_type.column,
                result_type.end_line,
                result_type.end_column,
            ).strip()
            monad_type = type_string.split()[0]
        else:
            print()
            print(
                "Warning : This definition has no signature type, which is ambiguous."
            )
            print("Since it has interactive parameters, we will assume")
            print("it is of type tactic unit.")
            print(lean_file.slice_string(ast.line, 0, ast.end_line, ast.end_column))
            print()
            type_string = None
            monad_type = "tactic"

        if monad_type in ["lean.parser", "parser", "smt_tactic", "old_conv"]:
            # user commands, macros, smt tactics, and old converstions
            # don't have proof tracing currently
            return None

        if monad_type == "itactic":
            print()
            print("Warning: This definition has type itactic, which is ambiguous.")
            print("Since it has interactive parameters, we will assume")
            print("it is of type tactic unit.")
            print(lean_file.slice_string(ast.line, 0, ast.end_line, ast.end_column))
            print()
            monad_type = "tactic"

        if monad_type not in ["conv", "tactic"]:
            print(f"Warning: This definition has an unknown type: {type_string}.")
            print(f"Warning: We will not modify it.")
            print(lean_file.slice_string(ast.line, 0, ast.end_line, ast.end_column))
            print()
            return None

        return monad_type

    def find_def_mod(
        self, lean_file: LeanFile, line: int, column: int
    ) -> Optional[Modification]:
        try:
            # parse the defintion
            parser = LeanParser(lean_file, line, column)
            parser.read_token("meta")
            parser.consume_space()
            if parser.is_token("def"):
                parser.read_token("def")
            elif parser.is_token("definition"):
                parser.read_token("definition")
            else:
                return None
            ast = parser.read_def()
        except:
            print()
            print(
                f"Warning: Definition parsing failed on line {line} of {lean_file.filename}."
            )
            print(lean_file.slice_string(line, 0, line + 1, 0))
            print("Skipping this and hoping it isn't an interactive tactic.")
            print()
            return None

        iparams = self.interactive_parameters(ast)
        if not iparams:
            return None

        monad_type = self.monad_type(lean_file, ast)
        if monad_type is None:
            return None

        end_line = ast.end_line + 1  # +1 since using the first line after that tactic

        tactic_name_path = ast.name.name_path
        if tactic_name_path[:2] == ["tactic", "."]:
            tactic_name_path = tactic_name_path[2:]
        if tactic_name_path[:2] == ["conv", "."]:
            tactic_name_path = tactic_name_path[2:]
        if tactic_name_path[:2] == ["interactive", "."]:
            tactic_name_path = tactic_name_path[2:]
        tactic_name = ".".join(tactic_name_path)

        return self.modify_def(
            lean_file, line, end_line, tactic_name, monad_type, iparams
        )

    def find_and_modify(self, file: Path, dryrun=bool):
        lean_file = LeanFile(str(file))
        modifications: List[Modification] = []
        for tokens in lean_file.find_pattern([TokenType.WHITESPACE, "meta"]):
            mod = self.find_def_mod(lean_file, tokens[-1].line, tokens[-1].column)
            if mod is not None:
                modifications.append(mod)
        if modifications and file not in DO_NOT_MODIFY:
            print(f"Modifying parameters in {file}.")
            modifier = LeanModifier(file)
            for mod in modifications:
                modifier.replace_lines(mod.line, mod.end_line, mod.new_lines)
            modifier.build_file(dryrun=dryrun)

            # print(file)
            # for mod in modifications:
            #     print("=======")
            #     print(lean_file.slice_string(mod.line, 0, mod.end_line, 0), end="")
            #     print(" => ")
            #     print(mod.new_lines, end="")
            #     print("=======")


def modify_interactive_tactic_parameters(dryrun: bool):
    for lean_dir in LEAN_DIRS:
        for f in lean_dir.glob("**/*.lean"):
            ModifyInterativeParameters().find_and_modify(f, dryrun=dryrun)


def _parse_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--sexp", action="store_true")
    return parser.parse_args()


def main():
    opts = _parse_main()
    dryrun = opts.dryrun
    if dryrun:
        print("[insert_proof_recording_code] Dry run. No files will be modified.")
    sexp = opts.sexp

    print("Insert tactic tracing code")
    insert_tactic_tracing_code(dryrun=dryrun, sexp=sexp)

    print("Insert tactic parameter tracing code")
    insert_param_tracing_code(dryrun=dryrun)

    print("Modify tactic parameters")
    modify_interactive_tactic_parameters(dryrun=dryrun)


if __name__ == "__main__":
    main()
