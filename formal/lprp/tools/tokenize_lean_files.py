# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tools for parsing lean files into tokens
"""
import collections
import dataclasses
import enum
from typing import Generator, List, Union, Optional


class TokenType(enum.Enum):
    LINE_COMMENT = 1
    BLOCK_COMMENT = 2
    WHITESPACE = 3
    SYMBOL = 4
    ALPHANUMERIC = 5
    STRING_LITERAL = 6
    BOF = 7  # Beginning of file
    EOF = 8  # End of file


class TerminalType(enum.Enum):
    """
    Two types only used for intermediate calculations
    """

    BLOCK_COMMENT_TERMINAL = -1
    STRING_LITERAL_TERMINAL = -2


# here we use 0-indexing for the lines and columns
@dataclasses.dataclass
class Token:
    """A token in the code, with position information"""

    string: str
    type: TokenType
    line: int
    # end_line: int
    column: int
    end_column: int


# there are other tokens as well, but these are important and/or easy to work with
SPECIAL_TOKENS_PARTS = [
    ["`", "["],
    ["(", ":", ":", ")"],
    [".", ".", "."],
    ["[", "whnf", "]"],
    ["<", "|", ">"],
    ["%", "%"],
    ["(", ")"],
    ["{", "!"],
    ["!", "}"],
    ["Type", "*"],
    ["Sort", "*"],
    ["(", ":"],
    [":", ")"],
    ["/", "/"],
    [".", "."],
    [":", "="],
    ["@", "@"],
    ["-", ">"],
    ["<", "-"],
    ["^", "."],
    ["@", "["],
    ["#", "check"],
    ["#", "reduce"],
    ["#", "eval"],
    ["#", "print"],
    ["#", "help"],
    ["#", "exit"],
    ["#", "compile"],
    ["#", "unify"],
    ["(", "|"],
    ["|", ")"],
    ["list", "Σ"],
    ["list", "Π"],
    ["⋂", "₀"],
]
# make sure they are sorted longest first
SPECIAL_TOKENS = list(sorted(SPECIAL_TOKENS_PARTS, key=len, reverse=True))


class LeanFile:
    filename: str  # useful for debugging
    lines: List[List[Token]]

    def __init__(self, filename: str):
        self.filename = filename
        assert filename.endswith(".lean"), filename

        # read file
        lines = []
        with open(filename, "r") as f:
            for line in f:
                lines.append(line)

        # tokenize
        self.lines = []
        prev_token_type = TokenType.BOF
        for i, line in enumerate(lines):
            tokens = LeanFile.tokenize_line(line, i, prev_token_type)
            prev_token_type = tokens[-1].type
            self.lines.append(tokens)
        final_token = self.lines[-1][-1]
        eof = Token(
            string="",
            type=TokenType.EOF,
            line=final_token.line,
            column=final_token.end_column,
            end_column=final_token.end_column,
        )
        self.lines[-1].append(eof)

    # this isn't completely accurate.  I think it also will capture
    # - char literals, e.g. 'a'
    # -
    # see https://github.com/leanprover/lean/blob/master/src/util/name.cpp
    @staticmethod
    def is_name_char(c: str):
        # special good characters
        if c in ["_", "'"]:
            return True
        # three special bad unicode
        if c in ["λ", "Π", "Σ"]:
            return False
        # standard characters
        u = ord(c)
        if (
            ord("A") <= u <= ord("Z")
            or ord("a") <= u <= ord("z")
            or ord("0") <= u <= ord("9")
        ):
            return True
        # other acceptable unicode
        if (
            (0x3B1 <= u and u <= 0x3C9 and u != 0x3BB)
            or (  # Lower greek, but lambda
                0x391 <= u and u <= 0x3A9 and u != 0x3A0 and u != 0x3A3
            )
            or (0x3CA <= u and u <= 0x3FB)  # Upper greek, but Pi and Sigma
            or (0x1F00 <= u and u <= 0x1FFE)  # Coptic letters
            or (0x2100 <= u and u <= 0x214F)  # Polytonic Greek Extended Character Set
            or (0x1D49C <= u and u <= 0x1D59F)  # Letter like block
            or (  # Latin letters, Script, Double-struck, Fractur
                0x207F <= u and u <= 0x2089
            )
            or (0x2090 <= u and u <= 0x209C)  # n superscript and numberic subscripts
            or (0x1D62 <= u and u <= 0x1D6A)  # letter-like subscripts
        ):  # letter-like subscripts
            return True
        return False

    @staticmethod
    def tokenize_line(
        line: str, line_num: int, prev_token_type: TokenType
    ) -> List[Token]:
        assert prev_token_type in [
            TokenType.BOF,
            TokenType.WHITESPACE,
            TokenType.BLOCK_COMMENT,
            TokenType.LINE_COMMENT,
            TokenType.STRING_LITERAL,
        ], (line, line_num, prev_token_type)
        # step 1: label char types
        prev_char = None
        prev_char_type = (
            prev_token_type
            if prev_token_type != TokenType.LINE_COMMENT
            else TokenType.WHITESPACE
        )
        char_types = []
        for i, char in enumerate(line):
            # label symbol type
            if prev_char_type == TokenType.LINE_COMMENT:
                char_type = TokenType.LINE_COMMENT
            elif prev_char_type == TokenType.BLOCK_COMMENT:
                if (prev_char, char) == ("-", "/"):
                    char_type = TerminalType.BLOCK_COMMENT_TERMINAL
                else:
                    char_type = TokenType.BLOCK_COMMENT
            elif prev_char_type == TokenType.STRING_LITERAL:
                if char == '"' and prev_char != "\\":
                    char_type = TerminalType.STRING_LITERAL_TERMINAL
                else:
                    char_type = TokenType.STRING_LITERAL
            elif char == "-" and line[i + 1] == "-":
                char_type = TokenType.LINE_COMMENT
            elif char == "/" and line[i + 1] == "-":
                char_type = TokenType.BLOCK_COMMENT
            elif char == '"':
                char_type = TokenType.STRING_LITERAL
            elif char.isspace():
                char_type = TokenType.WHITESPACE
            # this captures character literals too which is fine for now
            elif LeanFile.is_name_char(char):
                char_type = TokenType.ALPHANUMERIC
            else:
                char_type = TokenType.SYMBOL

            char_types.append(char_type)
            prev_char_type = char_type
            prev_char = char

        # step 2: join certain char_type pairs
        tokens = []

        def get_token_type(char_type: Union[TokenType, TerminalType]) -> TokenType:
            if char_type == TerminalType.BLOCK_COMMENT_TERMINAL:
                return TokenType.BLOCK_COMMENT
            if char_type == TerminalType.STRING_LITERAL_TERMINAL:
                return TokenType.STRING_LITERAL
            assert isinstance(char_type, TokenType)
            return char_type

        token_type = get_token_type(char_types[0])
        token_string = line[0]
        token_start = 0
        i = 0
        prev_char_type = char_types[0]

        for char, char_type in zip(line[1:], char_types[1:]):
            i += 1
            if (
                prev_char_type == TokenType.BLOCK_COMMENT
                and char_type == TerminalType.BLOCK_COMMENT_TERMINAL
            ):
                token_string += char
            elif (
                prev_char_type == TokenType.STRING_LITERAL
                and char_type == TerminalType.STRING_LITERAL_TERMINAL
            ):
                token_string += char
            elif char_type != TokenType.SYMBOL and prev_char_type == char_type:
                token_string += char
            else:
                assert i == len(token_string) + token_start, (
                    token_start,
                    i,
                    token_string,
                )
                tokens.append(Token(token_string, token_type, line_num, token_start, i))
                token_string = char
                token_type = get_token_type(char_type)
                token_start = i
            prev_char_type = char_type
        assert len(line) == len(token_string) + token_start, (
            token_start,
            len(line),
            token_string,
        )
        tokens.append(Token(token_string, token_type, line_num, token_start, len(line)))

        # step 3: combine certain tokens
        combined_tokens = []
        i = 0
        while i < len(tokens):
            # find longest token group which is in the list of special tokens
            # the tokens are sorted longest first
            for special in SPECIAL_TOKENS:
                if all(
                    t.string == s for t, s in zip(tokens[i : i + len(special)], special)
                ):
                    new_token = Token(
                        string="".join(special),
                        type=TokenType.SYMBOL,
                        line=tokens[i].line,
                        column=tokens[i].column,
                        end_column=tokens[i + len(special) - 1].end_column,
                    )
                    assert (
                        new_token.end_column == len(new_token.string) + new_token.column
                    ), (new_token.column, new_token.end_column, new_token.string)
                    i2 = i + len(special)
                    break
            else:
                new_token = tokens[i]
                i2 = i + 1

            combined_tokens.append(new_token)
            i = i2

        return combined_tokens

    def get_token(self, line: int, column: int) -> Token:
        assert line < len(self.lines)
        for token in self.lines[line]:
            if token.column < column:
                assert (
                    token.end_column <= column
                ), "The provided line/column ({},{}) doesn't match with the closest token ({},{}):\n{}".format(
                    line, column, line, token.column, token.string
                )
                continue
            if token.column == column:
                return token

        assert False

    def get_token_pos(self, line: int, column: int) -> int:
        assert line < len(self.lines)
        assert column <= self.lines[line][-1].column, (column, self.lines[line])
        for pos, token in enumerate(self.lines[line]):
            if token.column < column:
                continue
            if token.column == column:
                return pos
            assert False
        assert False

    def iter_tokens_right(
        self, start_line: int, start_column: int
    ) -> Generator[Token, None, None]:
        start_pos = self.get_token_pos(start_line, start_column)
        for line in self.lines[start_line:]:
            for token in line[start_pos:]:
                yield token
            start_pos = 0

    def iter_tokens_left(
        self, start_line: int, start_column: int
    ) -> Generator[Token, None, None]:
        start_pos = self.get_token_pos(start_line, start_column)
        for line in reversed(self.lines[0 : start_line + 1]):
            for token in reversed(line[0:start_pos]):  # don't include current token
                yield token
            start_pos = None

    def slice_tokens(
        self, start_line: int, start_column: int, end_line: int, end_column: int
    ) -> List[Token]:
        return [
            t
            for t in self.iter_tokens_right(start_line, start_column)
            if (t.line, t.column) < (end_line, end_column)
        ]

    def slice_string(
        self,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        clean: bool = False,
    ) -> str:
        tokens = self.slice_tokens(start_line, start_column, end_line, end_column)
        if clean:
            # remove comments, newlines, and extra whitespace
            strings = []
            in_whitespace = True
            for t in tokens:
                if t.type in (
                    TokenType.WHITESPACE,
                    TokenType.LINE_COMMENT,
                    TokenType.BLOCK_COMMENT,
                ):
                    if in_whitespace is False:
                        strings.append(" ")
                        in_whitespace = True
                else:
                    strings.append(t.string)
                    in_whitespace = False
            return "".join(strings).strip()
        else:
            return "".join(t.string for t in tokens)

    @staticmethod
    def token_matches_pattern(t: Token, p: Union[str, TokenType]):
        if isinstance(p, str):
            return t.string == p
        else:
            return t.type == p

    def find_pattern(
        self, pattern: List[Union[str, TokenType]]
    ) -> Generator[List[Token], None, None]:
        """
        Search for a pattern.

        pattern: A list of either strings (to match to tokens exactly)
                 or token types (also matching exactly to tokens).
        """

        tokens = collections.deque([])
        for t in self.iter_tokens_right(0, 0):
            if len(tokens) < len(pattern):
                tokens.append(t)
            else:
                tokens.append(t)
                tokens.popleft()
                if all(
                    self.token_matches_pattern(t, p) for t, p in zip(tokens, pattern)
                ):
                    yield list(tokens)

    def get_prev_matching_pattern(
        self, line: int, column: int, patterns: List[Union[str, TokenType]]
    ) -> Optional[Token]:
        for t in self.iter_tokens_left(line, column):
            if any(self.token_matches_pattern(t, p) for p in patterns):
                return t
        return None

    def get_next_matching_pattern(
        self, line: int, column: int, patterns: List[Union[str, TokenType]]
    ) -> Optional[Token]:
        for t in self.iter_tokens_right(line, column):
            if any(self.token_matches_pattern(t, p) for p in patterns):
                return t
        return None

    def find_left_bracket(
        self, lefts: List[str], rights: List[str], line: int, column: int
    ) -> Optional[Token]:
        stack_depth = (
            1  # iter_tokens_left skips the current token, so start counter at 1
        )
        for t in self.iter_tokens_left(line, column):
            if any(self.token_matches_pattern(t, l) for l in lefts):
                stack_depth -= 1
                if not stack_depth:
                    return t
            elif any(self.token_matches_pattern(t, l) for l in rights):
                stack_depth += 1
        return None

    def find_right_bracket(
        self, lefts: List[str], rights: List[str], line: int, column: int
    ) -> Optional[Token]:
        stack_depth = 0
        for t in self.iter_tokens_left(line, column):
            if any(self.token_matches_pattern(t, l) for l in rights):
                stack_depth -= 1
                if not stack_depth:
                    return t
            elif any(self.token_matches_pattern(t, l) for l in lefts):
                stack_depth += 1
        return None


if __name__ == "__main__":
    # filename = "tmp/proof_recording_example.lean"
    filename = "/Users/jasonrute/.elan/toolchains/leanprover-community-lean-3.20.0/lib/lean/library/init/data/nat/bitwise.lean"

    assert not LeanFile.is_name_char("λ")
    assert not LeanFile.is_name_char("Π")
    assert not LeanFile.is_name_char("Σ")

    leanfile = LeanFile(filename)
    print(len(leanfile.lines))
    for line in leanfile.lines:
        # print(line)
        print([t.string for t in line])
        # print([t.type for t in line])
        # print([t.start_column for t in line])
        # print([t.end_column for t in line])
