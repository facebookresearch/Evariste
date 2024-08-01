# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .coq_utils import remove_comments, split_commands


TESTS_REMOVE_COMMENTS = []

TESTS_REMOVE_COMMENTS.append(
    (
        r"""Variable rT : finType. (* Most definitions require a finType structure on rT *)
Implicit Type to : action D rT.""",
        r"""Variable rT : finType. 
Implicit Type to : action D rT.""",
    )
)

TESTS_REMOVE_COMMENTS.append(
    (
        r"""Coercion Formula : formula >-> type.

(* Declare (implicitly) the argument scope tags. *)
Notation "1" := Idx : group_presentation.""",
        r"""Coercion Formula : formula >-> type.


Notation "1" := Idx : group_presentation.""",
    )
)

TESTS_REMOVE_COMMENTS.append(
    (
        r"""Coercion Formula : formula >-> type.

(* Declare (implicitly) the argument scope tags.
Notation "1" := Idx : group_presentation.
*) bla""",
        r"""Coercion Formula : formula >-> type.

 bla""",
    )
)

TESTS_REMOVE_COMMENTS.append(
    (
        r"""now we are in a string "(* so we should not remove this *)" and keep it""",
        r"""now we are in a string "(* so we should not remove this *)" and keep it""",
    )
)

TESTS_REMOVE_COMMENTS.append(
    (
        r"""now we are in a string "(* so we should (* not *) remove this *)" and keep it""",
        r"""now we are in a string "(* so we should (* not *) remove this *)" and keep it""",
    )
)

TESTS_REMOVE_COMMENTS.append(
    (
        r"""now we are in a string "(*" so we should (* not *) remove this "*)" and keep it""",
        r"""now we are in a string "(*" so we should  remove this "*)" and keep it""",
    )
)

TESTS_REMOVE_COMMENTS.append(
    (
        r"""this is nested and should all be removed (* 1 (* 2 *) 3 *)""",
        r"""this is nested and should all be removed """,
    )
)

TESTS_REMOVE_COMMENTS.append(
    (
        r"""Definition Cn : term := (.\"n"; (!Nth (!CC "n") "n") !I !(penc (0,0))).""",
        r"""Definition Cn : term := (.\"n"; (!Nth (!CC "n") "n") !I !(penc (0,0))).""",
    )
)

TESTS_REMOVE_COMMENTS.append(
    (
        r"""(*Definition Cn : term := (.\"n"; (!Nth (!CC "n") "n") !I !(penc (0,0))).*)""",
        r"",
    )
)

TESTS_REMOVE_COMMENTS.append((r"""(* " *)""", r""))

TESTS_REMOVE_COMMENTS.append(
    (
        r"""(* -*- coq-prog-args: ("-emacs-U" "-R" "../monads" "); compile-command: "./makedoc.sh" -*- *)""",
        r"",
    )
)


TESTS_SPLIT_COMMANDS = []

TESTS_SPLIT_COMMANDS.append(
    (
        r"""Definition pair_invr x :=
  if x \is a pair_unitr then (x.1^-1, x.2^-1) else x.""",
        [
            r"""Definition pair_invr x := if x \is a pair_unitr then (x.1^-1, x.2^-1) else x."""
        ],
    )
)

TESTS_SPLIT_COMMANDS.append(
    (
        r"""Notation "[ 'zmodType' 'of' T 'for' cT ]" := (@clone T cT _ idfun)
  (at level 0, format "[ 'zmodType'  'of'  T  'for'  cT ]") : form_scope.""",
        [
            r"""Notation "[ 'zmodType' 'of' T 'for' cT ]" := (@clone T cT _ idfun) (at level 0, format "[ 'zmodType' 'of' T 'for' cT ]") : form_scope."""
        ],
    )
)

TESTS_SPLIT_COMMANDS.append(
    (
        r"""Lemma scalemx_eq0 m n a (A : 'M[R]_(m, n)) :
  (a *: A == 0) = (a == 0) || (A == 0).
Proof.   case nz_a: (a == 0) / eqP => [-> | _]; first by rewrite scale0r eqxx.
apply/eqP/eqP=> [aA0 | ->]; last exact: scaler0. apply/matrixP=> i j; apply/eqP; move/matrixP/(_ i j)/eqP: aA0.
by rewrite !mxE mulf_eq0 nz_a.

Qed.

""",
        [
            r"""Lemma scalemx_eq0 m n a (A : 'M[R]_(m, n)) : (a *: A == 0) = (a == 0) || (A == 0).""",
            r"""Proof.""",
            r"""case nz_a: (a == 0) / eqP => [-> | _]; first by rewrite scale0r eqxx.""",
            r"""apply/eqP/eqP=> [aA0 | ->]; last exact: scaler0.""",
            r"""apply/matrixP=> i j; apply/eqP; move/matrixP/(_ i j)/eqP: aA0.""",
            r"""by rewrite !mxE mulf_eq0 nz_a.""",
            r"""Qed.""",
        ],
    )
)


#
# test remove comments
#

n_fail = 0
for i, (x, y) in enumerate(TESTS_REMOVE_COMMENTS):
    y_ = remove_comments(x)
    if y_ == y:
        continue
    line_diff = [
        j
        for j, (line, line_) in enumerate(zip(y.split("\n"), y_.split("\n")))
        if line != line_
    ][0]
    n_fail += 1
    print(
        f"Failure in test {i + 1}/{len(TESTS_REMOVE_COMMENTS)}. Difference in line {line_diff}. "
        f"Expected:\n==========\n{y}\n==========\nbut found:\n==========\n{y_}\n=========="
    )

if n_fail == 0:
    print(f'All {len(TESTS_REMOVE_COMMENTS)} "remove_comments" tests ran successfully!')
else:
    print(f'Failed on {n_fail}/{len(TESTS_REMOVE_COMMENTS)} "remove_comments" tests!')


#
# test split commands
#

n_fail = 0
for i, (x, y) in enumerate(TESTS_SPLIT_COMMANDS):
    y_ = split_commands(x)
    if y_ == y:
        continue
    line_diff = [j for j, (line, line_) in enumerate(zip(y, y_)) if line != line_][0]
    n_fail += 1
    print(
        f"Failure in test {i + 1}/{len(TESTS_SPLIT_COMMANDS)}. Difference in line {line_diff}. "
        f"Expected:\n==========\n{y}\n==========\nbut found:\n==========\n{y_}\n=========="
    )

if n_fail == 0:
    print(f'All {len(TESTS_SPLIT_COMMANDS)} "split_commands" tests ran successfully!')
else:
    print(f'Failed on {n_fail}/{len(TESTS_SPLIT_COMMANDS)} "split_commands" tests!')
