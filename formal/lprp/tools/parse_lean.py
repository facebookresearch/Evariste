# Copyright (c) Meta Platforms, Inc. and affiliates.

import dataclasses
import json
from pathlib import Path
from pprint import pprint
import sys
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tokenize_lean_files import LeanFile, Token, TokenType

COMMANDS = {
    "theorem",
    "axiom",
    "axioms",
    "variable",
    "protected",
    "private",
    "hide",
    "definition",
    "meta",
    "mutual",
    "example",
    "noncomputable",
    "abbreviation",
    "variables",
    "parameter",
    "parameters",
    "constant",
    "constants",
    "using_well_founded",
    "[whnf]",
    "end",
    "namespace",
    "section",
    "prelude",
    "import",
    "inductive",
    "coinductive",
    "structure",
    "class",
    "universe",
    "universes",
    "local",
    "precedence",
    "reserve",
    "infixl",
    "infixr",
    "infix",
    "postfix",
    "prefix",
    "notation",
    "set_option",
    "open",
    "export",
    "@[",
    "attribute",
    "instance",
    "include",
    "omit",
    "init_quotient",
    "declare_trace",
    "add_key_equivalence",
    "run_cmd",
    "#check",
    "#reduce",
    "#eval",
    "#print",
    "#help",
    "#exit",
    "#compile",
    "#unify",
    "lemma",
    "def",
    "alias",
    "open_locale",
    "add_decl_doc",
    "library_note",
    "localized",
    "add_hint_tactic",
}

ARROWS = {"->", "→"}
BINDERS = {
    "assume",
    "Pi",
    "forall",
    "∀",
    "Π",
    "Σ",
    "∑",
    "∏",
    "exists",
    "∃",
    "λ",
    "fun",
    "⨆",
    "⋂",
    "⋃",
    "∫",
    "∫⁻",
    "⨁",
}
BRACKETS = {"{": "}", "(": ")", "[": "]", "⟨": "⟩", "⟪": "⟫", "⟮": "⟯", "`[": "]"}
LEFT_BRACKETS = set(BRACKETS.keys())
RIGHT_BRACKETS = set(BRACKETS.values())


def next_token(lean_file: LeanFile, token: Token):
    # note we use end_column here
    for t in lean_file.iter_tokens_right(token.line, token.end_column):
        return t


def prev_token(lean_file: LeanFile, token: Token):
    for t in lean_file.iter_tokens_left(token.line, token.column):
        return t


def search_right(lean_file: LeanFile, start: Token, targets: List[str]):
    for t in lean_file.iter_tokens_right(start.line, start.column):
        if t.string in targets:
            return t
    return None


def search_left(lean_file: LeanFile, start: Token, targets: List[str]):
    for t in lean_file.iter_tokens_left(start.line, start.column):
        if t.string in targets:
            return t
    return None


class AST:
    @dataclasses.dataclass
    class ASTData:
        line: int
        column: int
        end_line: int
        end_column: int

    @dataclasses.dataclass
    class Univs(ASTData):
        univs: List[str]

    @dataclasses.dataclass
    class Name(ASTData):
        name_path: List[str]

    @dataclasses.dataclass
    class ExprPart(ASTData):
        pass

    @dataclasses.dataclass
    class BoundExprPart(ExprPart):
        binder: str
        bound_part: "AST.Expr"
        expr: "AST.Expr"

    @dataclasses.dataclass
    class BracketExprPart(ExprPart):
        brackets: Tuple[str, str]
        exprs: List["AST.Expr"]

    @dataclasses.dataclass
    class ITEExprPart(ExprPart):
        if_expr: "AST.Expr"
        then_expr: "AST.Expr"
        else_expr: "AST.Expr"

    @dataclasses.dataclass
    class LetExprPart(ExprPart):
        var: "AST.Expr"
        expr: "AST.Expr"
        body: "AST.Expr"

    @dataclasses.dataclass
    class MatchCase(ASTData):
        pattern: "AST.Expr"
        expr: "AST.Expr"

    @dataclasses.dataclass
    class MatchExprPart(ExprPart):
        match_expr: "AST.Expr"
        cases: List["AST.MatchCase"]

    @dataclasses.dataclass
    class BeginExprPart(ExprPart):
        proof: "AST.BeginProof"

    @dataclasses.dataclass
    class CalcExprPart(ExprPart):
        parts: List["AST.Expr"]

    @dataclasses.dataclass
    class DoExprPart(ExprPart):
        parts: List["AST.Expr"]

    @dataclasses.dataclass
    class ByExprPart(ExprPart):
        proof: "AST.ByProof"

    @dataclasses.dataclass
    class TokenExprPart(ExprPart):
        t_string: str

    @dataclasses.dataclass
    class Expr(ASTData):
        expr_parts: List["AST.ExprPart"]

    @dataclasses.dataclass
    class ParamBlock(ASTData):
        brackets: Tuple[str, str]
        vars: Optional["AST.Expr"]
        type_expr: "AST.Expr"
        default_expr: Optional["AST.Expr"]

    @dataclasses.dataclass
    class SignatureType(ASTData):
        arg_types: List["AST.Expr"]
        result_type: "AST.Expr"

    @dataclasses.dataclass
    class Signature(ASTData):
        params: List["AST.ParamBlock"]
        signature_type: Optional["AST.SignatureType"]

    @dataclasses.dataclass
    class Body(ASTData):
        expr: "AST.Expr"

    @dataclasses.dataclass
    class DefLike(ASTData):
        univ: Optional["AST.Univs"]
        name: "AST.Name"
        signature: "AST.Signature"
        # body: 'AST.Body'

    @dataclasses.dataclass
    class TacticParam(ASTData):
        pass

    @dataclasses.dataclass
    class ExprTacticParam(TacticParam):
        pass

    @dataclasses.dataclass
    class ITacticTacticParam(TacticParam):
        tactic: "AST.ITactic"

    @dataclasses.dataclass
    class Tactic(ASTData):
        pass

    @dataclasses.dataclass
    class NamedTactic(Tactic):
        tactic_name: "AST.Name"
        args: List["AST.TacticParam"]

    @dataclasses.dataclass
    class CalcTactic(Tactic):
        pass

    @dataclasses.dataclass
    class Solve1Tactic(Tactic):
        tactics: List["AST.Tactic"]

    @dataclasses.dataclass
    class ITactic(Tactic):
        tactics: List["AST.Tactic"]

    @dataclasses.dataclass
    class SemicolonTactic(Tactic):
        tactic1: "AST.Tactic"
        tactic2: "AST.Tactic"
        first_semicolon_line: int
        first_semicolon_column: int
        semicolon_line: int
        semicolon_column: int
        semicolon_count: int

    @dataclasses.dataclass
    class SemicolonListTactic(Tactic):
        tactic1: "AST.Tactic"
        tactic_list: List["AST.Tactic"]
        first_semicolon_line: int
        first_semicolon_column: int
        semicolon_line: int
        semicolon_column: int
        semicolon_count: int

    @dataclasses.dataclass
    class AlternativeTactic(Tactic):
        tactic1: "AST.Tactic"
        tactic2: "AST.Tactic"
        alternative_line: int
        alternative_column: int

    @dataclasses.dataclass
    class ByProof(ASTData):
        tactic: "AST.Tactic"

    @dataclasses.dataclass
    class BeginProof(ASTData):
        tactics: List["AST.Tactic"]

    @dataclasses.dataclass
    class BracketProof(ASTData):
        tactics: List["AST.Tactic"]


class LeanParser:
    lean_file: LeanFile
    token_lines: List[List[Token]]
    line: int
    column: int
    pos: int
    current_token: Token
    parameter_positions: Dict[Tuple[int, int], List[Tuple[int, int]]]
    tactic_block_positions: Set[Tuple[int, int]]

    def __init__(
        self,
        lean_file: LeanFile,
        line: int,
        column: int,
        parameter_positions=None,
        tactic_block_positions=None,
    ):
        """
        Start parser on this file and this position
        """
        self.lean_file = lean_file
        self.token_lines = lean_file.lines
        self.line = line
        self.column = column
        pos = None
        current = None
        for i, t in enumerate(self.token_lines[self.line]):
            if t.column == column:
                pos = i
                current = t
        assert pos is not None
        assert current is not None
        self.pos = pos
        self.current_token = current
        self.parameter_positions = (
            {} if parameter_positions is None else parameter_positions
        )
        self.tactic_block_positions = (
            set() if tactic_block_positions is None else tactic_block_positions
        )

    # low level parser stuff
    def peek(self) -> Token:
        return self.current_token

    def next(self) -> Token:
        assert self.current_token.type != TokenType.EOF
        token = self.current_token

        # avoid using the absolute positions in the tokens
        length = self.current_token.end_column - self.current_token.column

        if self.pos + 1 < len(self.token_lines[self.line]):
            self.pos += 1
            self.column += length
        else:
            self.line += 1
            self.pos = 0
            self.column = 0

        self.current_token = self.token_lines[self.line][self.pos]
        assert self.current_token.line == self.line, (self.line, self.current_token)
        assert self.current_token.column == self.column, (
            self.column,
            self.current_token,
        )

        return token

    def is_eof(self) -> bool:
        return self.current_token.type == TokenType.EOF

    def format_file_location(self) -> str:
        line_str = "".join(t.string for t in self.token_lines[self.line])
        file_location = f"{self.lean_file.filename}:{self.line+1}:{self.column+1}"
        return f"{file_location}\n{self.line:04}: {line_str}      {' ' * self.column}{'^' * len(self.current_token.string)}"

    def raise_error(self, msg: str):
        raise Exception(f"{msg}:\n{self.format_file_location()}")

    # medium level parsing
    def start_pos(self) -> Tuple[int, int]:
        return (self.current_token.line, self.current_token.column)

    def end_pos(self) -> Tuple[int, int]:
        return (self.current_token.line, self.current_token.end_column)

    def read_next(self) -> str:
        if self.is_eof():
            self.raise_error("Expected token but EOF")
        return self.next().string

    def read_token(self, expected_string: str) -> str:
        t = self.next()
        if t.string != expected_string:
            self.raise_error(
                "Expected {} but found {}".format(repr(expected_string), repr(t.string))
            )
        return t.string

    def is_token(self, expected_string: str) -> bool:
        t = self.peek()
        return t.string == expected_string

    def read_token_in(self, expected_strings: Set[str]) -> str:
        t = self.next()
        if t.string not in expected_strings:
            self.raise_error(
                "Expected {} but found {}".format(
                    repr(expected_strings), repr(t.string)
                )
            )
        return t.string

    def is_token_in(self, expected_strings: Set[str]) -> bool:
        t = self.peek()
        return t.string in expected_strings

    def read_alphanum(self) -> str:
        t = self.next()
        if t.type != TokenType.ALPHANUMERIC:
            self.raise_error(
                "Expected alphanumeric but found {}".format(repr(t.string))
            )
        return t.string

    def is_alphanum(self) -> bool:
        t = self.peek()
        return t.type == TokenType.ALPHANUMERIC

    def consume_space(self):
        while not self.is_eof() and self.peek().type in [
            TokenType.WHITESPACE,
            TokenType.LINE_COMMENT,
            TokenType.BLOCK_COMMENT,
        ]:
            self.next()  # consume that token

    # high level parse stuff
    def read_univs(self) -> AST.Univs:
        line, column = self.start_pos()
        self.read_token("{")
        self.consume_space()
        univs = []
        while not self.is_token("}"):
            u = self.read_alphanum()
            univs.append(u)
            self.consume_space()
        self.read_token("}")
        self.consume_space()
        end_line, end_column = self.start_pos()
        return AST.Univs(
            univs=univs,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_name_part(self) -> str:
        if self.is_token("«"):
            l = self.read_token("«")
            n = self.read_alphanum()
            r = self.read_token("»")
            return l + n + r
        else:
            return self.read_alphanum()

    def read_namespace(self) -> str:
        name = self.read_name_part()
        self.read_token(".")
        return name

    def read_full_name(self) -> AST.Name:
        line, column = self.start_pos()
        name_path = []
        name = self.read_name_part()
        name_path.append(name)
        while self.is_token("."):
            self.read_token(".")
            name = self.read_name_part()
            name_path.append(name)
        end_line, end_column = self.start_pos()
        return AST.Name(
            name_path=name_path,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_expr_until(self, end_tokens: Set[str]) -> AST.Expr:
        """
        This is tricky, since I don't know the full expression syntax.
        I'll cheat and just match with the expected end token.
        """
        self.consume_space()
        expr_parts = []
        expr_line, expr_column = self.start_pos()
        while not self.is_token_in(end_tokens):
            # go through all the expression constructors
            # binders
            if self.is_token_in(BINDERS):
                line, column = self.start_pos()
                binder = self.read_token_in(BINDERS)
                bound_part = self.read_expr_until({","})
                self.read_token(",")
                expr = self.read_expr_until(end_tokens)
                end_line, end_column = self.start_pos()
                part = AST.BoundExprPart(
                    binder=binder,
                    bound_part=bound_part,  # not parsing this right now
                    expr=expr,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # brackets {} () [], etc with comma seperators
            elif self.is_token_in(LEFT_BRACKETS):
                line, column = self.start_pos()
                left_bracket = self.read_token_in(LEFT_BRACKETS)
                right_bracket = BRACKETS[left_bracket]
                sub_exprs = []
                sub_expr = self.read_expr_until({",", right_bracket})
                sub_exprs.append(sub_expr)
                while self.is_token(","):
                    self.read_token(",")
                    sub_expr = self.read_expr_until({",", right_bracket})
                    sub_exprs.append(sub_expr)
                self.read_token(right_bracket)
                end_line, end_column = self.start_pos()
                part = AST.BracketExprPart(
                    brackets=(left_bracket, right_bracket),
                    exprs=sub_exprs,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # if then else
            elif self.is_token("if"):
                line, column = self.start_pos()
                self.read_token("if")
                if_part = self.read_expr_until({"then"})
                self.read_token("then")
                then_part = self.read_expr_until({"else"})
                self.read_token("else")
                else_part = self.read_expr_until(end_tokens)
                end_line, end_column = self.start_pos()
                part = AST.ITEExprPart(
                    if_expr=if_part,
                    then_expr=then_part,
                    else_expr=else_part,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # let := in
            elif self.is_token("let"):
                line, column = self.start_pos()
                self.read_token("let")
                var_part = self.read_expr_until({":="})
                self.read_token(":=")
                expr_part = self.read_expr_until({"in"})
                self.read_token("in")
                body_part = self.read_expr_until(end_tokens)
                end_line, end_column = self.start_pos()
                part = AST.LetExprPart(
                    var=var_part,
                    expr=expr_part,
                    body=body_part,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # match with | := end
            elif self.is_token("match"):
                line, column = self.start_pos()
                self.read_token("match")
                match_part = self.read_expr_until({"with"})
                self.read_token("with")
                self.consume_space()
                cases = []
                first = True
                while not self.is_token("end"):
                    case_line, case_column = self.start_pos()
                    if not first:
                        self.read_token("|")
                    else:
                        first = False
                        if self.is_token("|"):
                            self.read_token("|")
                    self.consume_space()
                    case_start = self.read_expr_until({":="})
                    case_body = self.read_expr_until({"|", "end"})
                    case_end_line, case_end_column = self.start_pos()
                    cases.append(
                        AST.MatchCase(
                            pattern=case_start,
                            expr=case_body,
                            line=case_line,
                            column=case_column,
                            end_line=case_end_line,
                            end_column=case_end_column,
                        )
                    )
                self.read_token("end")
                end_line, end_column = self.start_pos()
                part = AST.MatchExprPart(
                    match_expr=match_part,
                    cases=cases,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # begin end
            elif self.is_token("begin"):
                line, column = self.start_pos()
                proof = self.read_begin()
                end_line, end_column = self.start_pos()
                part = AST.BeginExprPart(
                    proof=proof,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # calc
            elif self.is_token("calc"):
                line, column = self.start_pos()
                self.read_token("calc")
                calc_parts = []
                calc_part = self.read_expr_until({"..."} | end_tokens)
                calc_parts.append(calc_part)
                while self.is_token("..."):
                    self.read_token("...")
                    calc_part = self.read_expr_until({"..."} | end_tokens)
                    calc_parts.append(calc_part)
                end_line, end_column = self.start_pos()
                part = AST.CalcExprPart(
                    parts=calc_parts,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # do
            elif self.is_token("do"):
                line, column = self.start_pos()
                self.read_token("do")
                do_parts = []
                do_part = self.read_expr_until({","} | end_tokens)
                do_parts.append(do_part)
                while self.is_token(","):
                    self.read_token(",")
                    do_part = self.read_expr_until({","} | end_tokens)
                    do_parts.append(do_part)
                end_line, end_column = self.start_pos()
                part = AST.DoExprPart(
                    parts=do_parts,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # by
            elif self.is_token("by"):
                line, column = self.start_pos()
                proof = self.read_by()
                end_line, end_column = self.start_pos()
                part = AST.ByExprPart(
                    proof=proof,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            # unexpected token
            elif self.is_token_in(COMMANDS | RIGHT_BRACKETS):
                s = self.peek().string
                self.raise_error(
                    f"Expected expression to end {end_tokens} but found {repr(s)}"
                )

            else:
                # Include whitespace here since
                # it may be important, but convert commments
                # to whitespace
                # TODO: Remove comments
                line, column = self.start_pos()
                s = self.read_next()
                end_line, end_column = self.start_pos()
                part = AST.TokenExprPart(
                    t_string=s,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )

            expr_parts.append(part)
        expr_end_line, expr_end_column = self.start_pos()
        return AST.Expr(
            expr_parts=expr_parts,
            line=expr_line,
            column=expr_column,
            end_line=expr_end_line,
            end_column=expr_end_column,
        )

    def read_parameter_block(self) -> AST.ParamBlock:
        line, column = self.start_pos()
        left_bracket = self.read_token_in(LEFT_BRACKETS)
        right_bracket = BRACKETS[left_bracket]
        self.consume_space()
        # note a square bracket [ ] may not have variables or a colon
        expr = self.read_expr_until({":", ":=", right_bracket})

        if self.is_token(":"):
            self.read_token(":")
            vars = expr
            expr = self.read_expr_until({":=", right_bracket})
        else:
            vars = None

        if self.is_token(":="):
            self.read_token(":=")
            default_expr = self.read_expr_until({right_bracket})
        else:
            default_expr = None

        self.read_token(right_bracket)
        self.consume_space()
        end_line, end_column = self.start_pos()
        return AST.ParamBlock(
            brackets=(left_bracket, right_bracket),
            vars=vars,
            type_expr=expr,
            default_expr=default_expr,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_signature_type(self) -> AST.SignatureType:
        line, column = self.start_pos()
        arg_types = []
        type_expr = self.read_expr_until(ARROWS | {":=", "|"})
        while self.is_token_in(ARROWS):
            self.read_token_in(ARROWS)
            arg_types.append(type_expr)
            type_expr = self.read_expr_until(ARROWS | {":=", "|"})
        end_line, end_column = self.start_pos()
        return AST.SignatureType(
            arg_types=arg_types,
            result_type=type_expr,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_signature(self) -> AST.Signature:
        line, column = self.start_pos()
        params = []
        while self.is_token_in(LEFT_BRACKETS):
            param = self.read_parameter_block()
            params.append(param)
        self.consume_space()
        if self.is_token(":"):
            self.read_token(":")
            self.consume_space()
            sign_type = self.read_signature_type()
        else:
            sign_type = None
        end_line, end_column = self.start_pos()
        return AST.Signature(
            params=params,
            signature_type=sign_type,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_body(self) -> AST.Body:
        line, column = self.start_pos()
        expr = self.read_expr_until(COMMANDS)
        end_line, end_column = self.start_pos()
        return AST.Body(
            expr=expr,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_def(self) -> AST.DefLike:
        line, column = self.start_pos()
        self.consume_space()
        if self.is_token("{"):
            univs = self.read_univs()
        else:
            univs = None
        name = self.read_full_name()
        self.consume_space()
        signature = self.read_signature()
        # TODO: fix this
        # body = self.read_body()
        end_line, end_column = self.start_pos()
        return AST.DefLike(
            univ=univs,
            name=name,
            signature=signature,
            # body=body,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def consume_matching_brackets(self) -> None:
        self.read_token_in(LEFT_BRACKETS)
        # some left brackets like `[ have a right brack ]
        # which matches another left bracket.
        # The easiest thing is just to consider all brackets.
        depth = 1
        while True:
            if self.is_token_in(LEFT_BRACKETS):
                self.read_token_in(LEFT_BRACKETS)
                depth += 1
            elif self.is_token_in(RIGHT_BRACKETS):
                self.read_token_in(RIGHT_BRACKETS)
                depth -= 1
                if not depth:
                    return None
            else:
                self.read_next()

    def read_named_tactic(self) -> AST.NamedTactic:
        line, column = self.start_pos()
        tactic_name = self.read_full_name()
        self.consume_space()
        parameters = []
        visted_parameters = set()
        while True:
            if (
                self.start_pos() in self.parameter_positions
                and self.start_pos() not in visted_parameters
            ):
                param_line, param_column = self.start_pos()
                visted_parameters.add((param_line, param_column))
                for (param_end_line, param_end_column) in self.parameter_positions[
                    param_line, param_column
                ]:
                    if (
                        param_end_line,
                        param_end_column,
                    ) > self.start_pos() and self.start_pos() in self.tactic_block_positions:
                        # this is an itactic parameter
                        itactic = self.read_itactic()
                        self.consume_space()
                        assert (param_end_line, param_end_column) == self.start_pos()
                        parameters.append(
                            AST.ITacticTacticParam(
                                tactic=itactic,
                                line=param_line,
                                column=param_column,
                                end_line=param_end_line,
                                end_column=param_end_column,
                            )
                        )
                    else:
                        while self.start_pos() < (param_end_line, param_end_column):
                            self.read_next()
                        if self.start_pos() != (param_end_line, param_end_column):
                            self.raise_error(
                                f"End of parameter is in middle of a token.  Expected parameter to end at {(param_end_line, param_end_column)}"
                            )
                        # TODO: If a tactic parameter or tactic list parameter, then zoom into tactics???
                        parameters.append(
                            AST.TacticParam(
                                line=param_line,
                                column=param_column,
                                end_line=param_end_line,
                                end_column=param_end_column,
                            )
                        )
                        self.consume_space()
            elif self.start_pos() in self.tactic_block_positions:
                param_line, param_column = self.start_pos()
                itactic = self.read_itactic()
                self.consume_space()
                param_end_line, param_end_column = self.start_pos()
                parameters.append(
                    AST.ITacticTacticParam(
                        tactic=itactic,
                        line=param_line,
                        column=param_column,
                        end_line=param_end_line,
                        end_column=param_end_column,
                    )
                )
            elif self.is_token_in(LEFT_BRACKETS):
                # This could be:
                # - Optional config paramters {foo := tt}
                # - itactic environment for some strange monad
                # - `[...]
                param_line, param_column = self.start_pos()
                print(
                    "WARNING: Non-interactive parameter.  Check that this is not a parsing error."
                )
                print(self.format_file_location())
                self.consume_matching_brackets()
                self.consume_space()
                param_end_line, param_end_column = self.start_pos()
                parameters.append(
                    AST.TacticParam(
                        line=param_line,
                        column=param_column,
                        end_line=param_end_line,
                        end_column=param_end_column,
                    )
                )
            elif self.is_alphanum() and self.peek().string.isnumeric():
                param_line, param_column = self.start_pos()
                print(
                    "WARNING: Non-interactive parameter.  Check that this is not a parsing error."
                )
                print(self.format_file_location())
                self.read_alphanum()
                self.consume_space()
                param_end_line, param_end_column = self.start_pos()
                parameters.append(
                    AST.TacticParam(
                        line=param_line,
                        column=param_column,
                        end_line=param_end_line,
                        end_column=param_end_column,
                    )
                )
            elif self.is_token_in(COMMANDS | {"else", "in"}):
                break
            elif self.is_alphanum():
                param_line, param_column = self.start_pos()
                print(
                    "WARNING: Non-interactive parameter.  Check that this is not a parsing error."
                )
                print(self.format_file_location())
                self.read_full_name()
                self.consume_space()
                param_end_line, param_end_column = self.start_pos()
                parameters.append(
                    AST.TacticParam(
                        line=param_line,
                        column=param_column,
                        end_line=param_end_line,
                        end_column=param_end_column,
                    )
                )
            else:
                break
        end_line, end_column = self.start_pos()
        return AST.NamedTactic(
            tactic_name=tactic_name,
            args=parameters,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_single_tactic(
        self,
    ) -> Union[AST.NamedTactic, AST.Solve1Tactic, AST.CalcTactic]:
        # can be {}, or named tactic
        line, column = self.start_pos()
        if self.is_token_in({"{", "begin"}):
            t_list = self.read_tactic_list()
            self.consume_space()
            end_line, end_column = self.start_pos()
            return AST.Solve1Tactic(
                tactics=t_list,
                line=line,
                column=column,
                end_line=end_line,
                end_column=end_column,
            )
        elif self.is_token("by"):
            self.read_token("by")
            self.consume_space()
            tactic = self.read_maybe_semicolon_tactic()
            end_line, end_column = self.start_pos()
            return AST.Solve1Tactic(
                tactics=[tactic],
                line=line,
                column=column,
                end_line=end_line,
                end_column=end_column,
            )
        elif self.is_token("calc"):
            # do this to get the calc expression
            self.read_expr_until(
                {"end", ";", ",", "|", "<|>"} | RIGHT_BRACKETS | COMMANDS
            )
            self.consume_space()
            end_line, end_column = self.start_pos()
            return AST.CalcTactic(
                line=line, column=column, end_line=end_line, end_column=end_column
            )
        elif self.is_token("do"):
            self.raise_error(
                'Parsing "do" tactic in interactive mode not yet implemented'
            )
        else:
            return self.read_named_tactic()

    def read_maybe_alt_tactic(
        self,
    ) -> Union[
        AST.AlternativeTactic, AST.NamedTactic, AST.Solve1Tactic, AST.CalcTactic
    ]:
        # can be <|>, {}, or named tactic
        # alternatives group as done <|> (done <|> trivial)
        line, column = self.start_pos()
        first_tactic = self.read_single_tactic()
        if self.is_token("<|>"):
            alternative_line, alternative_column = self.start_pos()
            self.read_token("<|>")
            self.consume_space()
            second_tactic = self.read_maybe_alt_tactic()
            end_line, end_column = self.start_pos()
            return AST.AlternativeTactic(
                tactic1=first_tactic,
                tactic2=second_tactic,
                alternative_line=alternative_line,
                alternative_column=alternative_column,
                line=line,
                column=column,
                end_line=end_line,
                end_column=end_column,
            )
        else:
            return first_tactic

    def read_tactic_list(self) -> List[AST.Tactic]:
        left = self.read_next()
        if left not in ["begin", "{", "["]:
            self.raise_error('Expected "begin", "{", or "[".')
        right = {"begin": "end", "{": "}", "[": "]"}[left]
        tactics = []

        while True:
            self.consume_space()
            if self.is_token(right):
                break  # possible to have trailing ","
            t = self.read_maybe_semicolon_tactic()
            tactics.append(t)

            if self.is_token(","):
                self.read_token(",")
                continue
            elif self.is_token(right):
                break
            else:

                self.raise_error(f'Expected "," or "{right}"')
        self.read_token(right)
        self.consume_space()
        return tactics

    def read_maybe_semicolon_tactic(self) -> AST.Tactic:
        # can be ;, <|>, {}, or named tactic
        # semicolons group as (((skip ; skip) ; induction n) ; skip) ; simp
        line, column = self.start_pos()
        tactic = self.read_maybe_alt_tactic()
        first_semicolon_pos = None
        semicolon_count = 0
        while self.is_token(";"):
            semicolon_pos = self.start_pos()
            semicolon_count += 1
            if first_semicolon_pos is None:
                first_semicolon_pos = semicolon_pos
            self.read_token(";")
            self.consume_space()
            if self.is_token("["):
                t_list = self.read_tactic_list()
                self.consume_space
                end_line, end_column = self.start_pos()
                tactic = AST.SemicolonListTactic(
                    tactic1=tactic,
                    tactic_list=t_list,
                    first_semicolon_line=first_semicolon_pos[0],
                    first_semicolon_column=first_semicolon_pos[1],
                    semicolon_line=semicolon_pos[0],
                    semicolon_column=semicolon_pos[1],
                    semicolon_count=semicolon_count,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )
            else:
                tactic2 = self.read_maybe_alt_tactic()
                end_line, end_column = self.start_pos()
                tactic = AST.SemicolonTactic(
                    tactic1=tactic,
                    tactic2=tactic2,
                    first_semicolon_line=first_semicolon_pos[0],
                    first_semicolon_column=first_semicolon_pos[1],
                    semicolon_line=semicolon_pos[0],
                    semicolon_column=semicolon_pos[1],
                    semicolon_count=semicolon_count,
                    line=line,
                    column=column,
                    end_line=end_line,
                    end_column=end_column,
                )
        return tactic

    def read_by(self) -> AST.ByProof:
        line, column = self.start_pos()
        self.read_token("by")
        self.consume_space()
        tactic = self.read_maybe_semicolon_tactic()
        end_line, end_column = self.start_pos()
        return AST.ByProof(
            tactic=tactic,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_begin(self) -> AST.BeginProof:
        line, column = self.start_pos()
        if not self.is_token("begin"):
            self.raise_error('Expected "begin"')
        tactics = self.read_tactic_list()
        end_line, end_column = self.start_pos()
        return AST.BeginProof(
            tactics=tactics,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_bracket_proof(self) -> AST.BracketProof:
        line, column = self.start_pos()
        if not self.is_token("{"):
            self.raise_error('Expected "{"')
        tactics = self.read_tactic_list()
        end_line, end_column = self.start_pos()
        return AST.BracketProof(
            tactics=tactics,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )

    def read_itactic(self) -> AST.ITactic:
        line, column = self.start_pos()
        if not self.is_token_in({"{", "begin"}):
            self.raise_error('Expected "{" or "begin"')
        tactics = self.read_tactic_list()
        end_line, end_column = self.start_pos()
        return AST.ITactic(
            tactics=tactics,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
        )


def slice(lean_file, start, end):
    s = []
    for t in lean_file.iter_tokens_right(start.line, start.start_column):
        if t.line == end.line and t.start_column == end.start_column:
            return s
        s.append(t)

    return None


def main():
    assert len(sys.argv) == 2
    lean_dir = Path(sys.argv[1])
    assert lean_dir.is_dir

    for f in lean_dir.glob("**/*.lean"):
        # print(f)
        decls = set()
        lean_file = LeanFile(str(f))
        for i, line in enumerate(lean_file.lines):
            for token in line:
                if (
                    token.string == "def"
                    and prev_token(lean_file, token).type == TokenType.WHITESPACE
                ):
                    try:
                        parser = LeanParser(lean_file, token.line, token.column)
                        parser.read_token("def")
                        ast = parser.read_def()
                        # pprint(ast)

                        def expr_begins_with(expr: AST.Expr, tokens: List[str]) -> bool:
                            if len(expr.expr_parts) < len(tokens):
                                return False
                            for e, t in zip(expr.expr_parts, tokens):
                                if not isinstance(e, AST.TokenExprPart):
                                    return False
                                elif e.t_string != t:
                                    return False
                            return True

                        def extract_parser(param_expr: AST.Expr, start: int):
                            # strip leading whitepace and $
                            parts = param_expr.expr_parts
                            for i in range(start, len(parts)):
                                part = parts[i]
                                if isinstance(part, AST.TokenExprPart):
                                    if part.t_string == "$":
                                        continue
                                    if part.t_string.strip() == "":
                                        continue
                                    break
                                break
                            return {
                                "pos": (
                                    parts[0].line,
                                    parts[0].column,
                                    parts[-1].end_line,
                                    parts[-1].end_column,
                                ),
                                "command": "parse",
                                "parser_pos": (
                                    parts[i].line,
                                    parts[i].column,
                                    parts[-1].end_line,
                                    parts[-1].end_column,
                                ),
                            }

                        def process_param_expr(param_expr: AST.Expr):
                            if expr_begins_with(param_expr, ["parse"]):
                                return extract_parser(param_expr, 1)
                            elif expr_begins_with(
                                param_expr, ["interative", ".", "parse"]
                            ):
                                return extract_parser(param_expr, 3)
                            elif expr_begins_with(param_expr, ["itactic"]):
                                return {
                                    "pos": (
                                        param_expr.expr_parts[0].line,
                                        param_expr.expr_parts[0].column,
                                        param_expr.expr_parts[0].end_line,
                                        param_expr.expr_parts[0].end_column,
                                    ),
                                    "command": "itactic",
                                }
                            elif expr_begins_with(
                                param_expr, ["interactive", ",", "itactic"]
                            ):
                                return {
                                    "pos": (
                                        param_expr.expr_parts[0].line,
                                        param_expr.expr_parts[0].column,
                                        param_expr.expr_parts[2].end_line,
                                        param_expr.expr_parts[2].end_column,
                                    ),
                                    "command": "itactic",
                                }

                        interactive_parts = []
                        n = -1
                        for param in ast.signature.params:
                            n += 1
                            result = process_param_expr(param.type_expr)
                            if result is not None:
                                result["n"] = n
                                interactive_parts.append(result)
                        if ast.signature.signature_type is not None:
                            for param in ast.signature.signature_type.arg_types:
                                n += 1
                                result = process_param_expr(param)
                                if result is not None:
                                    result["n"] = n
                                    interactive_parts.append(result)
                        if interactive_parts:
                            print()
                            print(f)
                            print("".join(t.string for t in line))
                            # pprint(interactive_parts)
                            # reconstruct new declaration
                            s = ""
                            pos = (i, 0)
                            for part in interactive_parts:
                                s += lean_file.slice_string(
                                    pos[0], pos[1], part["pos"][0], part["pos"][1]
                                )
                                if part["command"] == "itactic":
                                    s += "interactive.pr.parse interactive.pr.itactic "
                                elif part["command"] == "parse":
                                    s += "interactive.pr.parse"
                                    parser = lean_file.slice_string(
                                        part["parser_pos"][0],
                                        part["parser_pos"][1],
                                        part["parser_pos"][2],
                                        part["parser_pos"][3],
                                    )
                                    parser = parser.strip()
                                    if parser.startswith("(") and parser.endswith(")"):
                                        parser = parser[1:-1].strip()
                                    s += " " + str(part["n"])
                                    s += " " + json.dumps(parser)
                                    s += " (" + parser + ") "
                                else:
                                    assert False
                                pos = (part["pos"][2], part["pos"][3])
                            s += lean_file.slice_string(
                                pos[0], pos[1], pos[0] + 1, 0
                            )  # Go to end of the line (including newline)
                            print(s)

                    except Exception:
                        print()
                        print(f)
                        print("".join(t.string for t in line))
                        traceback.print_exc()


if __name__ == "__main__":
    main()
