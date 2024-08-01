# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .hol_light_tokenizer import tokenize_hl, detokenize_hl


TESTS_TOKENIZER = []

# fmt: off
TESTS_TOKENIZER.append((
    r'g UNIV:(A)finite_image->bool = IMAGE finite_index (1..dimindex(:A));;',
    'g UNIV : ( A ) finite_image -> bool = IMAGE finite_index ( 1 .. dimindex ( : A ) ) ;;'
))

TESTS_TOKENIZER.append((
    r"'p' =q' '",
    r"' p @@@@' = q @@@@' '"
))

TESTS_TOKENIZER.append((
    r'g `!x y. x >= y <=> y <= x`;;',
    r'g ` ! x y . x >= y <=> y <= x ` ;;'
))

TESTS_TOKENIZER.append((
    r'aaa*                   --bb',
    r'aaa * -- bb'
))

TESTS_TOKENIZER.append((
    r'aaa*	--bb',
    r'aaa * -- bb'
))

TESTS_TOKENIZER.append((
    r"a__'9a''a*	--bb ok'",
    r"a__ @@@@' @@@@9a @@@@' @@@@' @@@@a * -- bb ok @@@@'"
))

TESTS_TOKENIZER.append((
    r"a_''_'a''a*	--bb o'k'",
    r"a_ @@@@' @@@@' @@@@_ @@@@' @@@@a @@@@' @@@@' @@@@a * -- bb o @@@@' @@@@k @@@@'"
))

TESTS_TOKENIZER.append((
    r'e (MP_TAC (ISPECL [parse_term "(:complex) DELETE z"; parse_term "g:real^1->complex"] PATH_INTEGRAL_NEARBY_ENDS));;',
    r'e ( MP_TAC ( ISPECL [ parse_term " ( : complex ) DELETE z " ; parse_term " g : real ^ 1 -> complex " ] PATH_INTEGRAL_NEARBY_ENDS ) ) ;;'
))

TESTS_TOKENIZER.append((
    r"e (EXISTS_TAC `{a:real^1,b}`);;",
    r'e ( EXISTS_TAC ` { a : real ^ 1 , b } ` ) ;;'
))

TESTS_TOKENIZER.append((
    r'e (MAP_EVERY EXISTS_TAC   [parse_term "s:complex->bool"; parse_term "{}:complex->bool"]);;',
    r'e ( MAP_EVERY EXISTS_TAC [ parse_term " s : complex -> bool " ; parse_term " { } : complex -> bool " ] ) ;;'
))

TESTS_TOKENIZER.append((
    r'e (SET_TAC []);;',
    r'e ( SET_TAC [ ] ) ;;'
))

TESTS_TOKENIZER.append((
    r'g `!s t. s <=_c t <=> ?g. !x. x IN s ==> ?y. y IN t /\ (g y = x)`;;',
    r'g ` ! s t . s <=_c t <=> ? g . ! x . x IN s ==> ? y . y IN t /\ ( g y = x ) ` ;;'
))

TESTS_TOKENIZER.append((
    r'g `!s:B->bool c:C. ({}:A->bool) ^_c s =_c if s = {} then {c} else {}`;;',
    r'g ` ! s : B -> bool c : C . ( { } : A -> bool ) ^_c s =_c if s = { } then { c } else { } ` ;;'
))

TESTS_TOKENIZER.append((
    r'^_c =_c <_c >_c <=_c >=_c +_c *_c ^_c',
    r'^_c =_c <_c >_c <=_c >=_c +_c *_c ^_c'
))

TESTS_TOKENIZER.append((
    r'2 + 33 + 44 = 79',
    r'2 + <INTEGER> 3 3 </INTEGER> + <INTEGER> 4 4 </INTEGER> = <INTEGER> 7 9 </INTEGER>'
))

TESTS_TOKENIZER.append((
    r'11.2 + 1.230 + 33.5',
    r'<DECIMAL> 1 1 . 2 </DECIMAL> + <DECIMAL> 1 . 2 3 0 </DECIMAL> + <DECIMAL> 3 3 . 5 </DECIMAL>'
))

TESTS_TOKENIZER.append((
    r'REAL_RAT_SUB_CONV `&355 / &113 - #3.1415926`',
    r'REAL_RAT_SUB_CONV ` & <INTEGER> 3 5 5 </INTEGER> / & <INTEGER> 1 1 3 </INTEGER> - # <DECIMAL> 3 . 1 4 1 5 9 2 6 </DECIMAL> `'
))
# fmt: on


TESTS_DETOKENIZER = []

# fmt: off
TESTS_DETOKENIZER.append((
    r"f''''",
    r"f @@@@' @@@@' @@@@' @@@@'"
))

TESTS_DETOKENIZER.append((
    r"f''A''",
    r"f @@@@' @@@@' @@@@A @@@@' @@@@'"
))

TESTS_DETOKENIZER.append((
    r"f''A''A",
    r"f @@@@' @@@@' @@@@A @@@@' @@@@' @@@@A"
))

TESTS_DETOKENIZER.append((
    r"a__'a''a ok'",
    r"a__ @@@@' @@@@a @@@@' @@@@' @@@@a ok @@@@'"
))

TESTS_DETOKENIZER.append((
    r"a_''_'a''a o'k'",
    r"a_ @@@@' @@@@' @@@@_ @@@@' @@@@a @@@@' @@@@' @@@@a o @@@@' @@@@k @@@@'"
))

TESTS_DETOKENIZER.append((
    r'g `! x y . x >= y <=> y <= x` ;;',
    r'g ` ! x y . x >= y <=> y <= x ` ;;'
))

TESTS_DETOKENIZER.append((
    r'^_c =_c <_c >_c <=_c >=_c +_c *_c ^_c',
    r'^_c =_c <_c >_c <=_c >=_c +_c *_c ^_c'
))

TESTS_DETOKENIZER.append((
    r'2 + 33 + 44 = 79',
    r'2 + <INTEGER> 3 3 </INTEGER> + <INTEGER> 4 4 </INTEGER> = <INTEGER> 7 9 </INTEGER>'
))

TESTS_DETOKENIZER.append((
    r'11.2 + 1.230 + 33.5',
    r'<DECIMAL> 1 1 . 2 </DECIMAL> + <DECIMAL> 1 . 2 3 0 </DECIMAL> + <DECIMAL> 3 3 . 5 </DECIMAL>'
))

TESTS_DETOKENIZER.append((
    r"derivative f = f'",
    r"derivative f = f @@@@'"
))

TESTS_DETOKENIZER.append((
    r"derivative f = f'''",
    r"derivative f = f @@@@' @@@@' @@@@'"
))
# fmt: on


#
# test tokenizer
#

n_fail = 0
for i, (x, y) in enumerate(TESTS_TOKENIZER):
    y_ = tokenize_hl(x)
    if y_ == y.split():
        continue
    n_fail += 1
    print(
        f"Failure in test {i + 1}/{len(TESTS_TOKENIZER)}. "
        f"Expected:\n==========\n{y.split()}\n==========\nbut found:\n==========\n{y_}\n=========="
    )

if n_fail == 0:
    print(f'All {len(TESTS_TOKENIZER)} "tokenize_hl" tests ran successfully!')
else:
    print(f'Failed on {n_fail}/{len(TESTS_TOKENIZER)} "tokenize_hl" tests!')


#
# test detokenizer
#

n_fail = 0
for i, (x, y) in enumerate(TESTS_DETOKENIZER):
    x_ = detokenize_hl(y.split())
    if x_ == x:
        continue
    n_fail += 1
    print(
        f"Failure in test {i + 1}/{len(TESTS_DETOKENIZER)}. "
        f"Expected:\n==========\n{x}\n==========\nbut found:\n==========\n{x_}\n=========="
    )

if n_fail == 0:
    print(f'All {len(TESTS_DETOKENIZER)} "detokenize_hl" tests ran successfully!')
else:
    print(f'Failed on {n_fail}/{len(TESTS_TOKENIZER)} "detokenize_hl" tests!')
