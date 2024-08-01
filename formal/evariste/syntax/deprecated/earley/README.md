### Using the parser
To build the cpp extension, run the command:

```python setup.py build_ext --inplace```

### The grammar
The grammar is automatically generated from the `export_grammar.py` utility.

All tokens in a line starting with `#Symbol#` will be tokenized into the terminal `#Symbol#`.
This is done for class_var, setvar_var and wff_var.

The remainder of the grammar is described with non-terminals starting with the `$` character. All other tokens are understood as terminals.

At the end of a rule, `$*rule_name` is understood as the name for this derivation and is used to build a parse-tree.


```
#CLASS_VAR# 0 1 2 3 4 5 6 7 8 9
#SET_VAR# a b c d
#WFF_VAR# ph ch ps
$start wff $wff
$start class $class
$start set $set
$start setvar $set
$class ( $class + $class ) $*wcadd
$class #CLASS_VAR# $*c1
```

### The code
The `Grammar` class holds all the code.
Hashable EarleyItems describe a parsing state as well as a pointer to a ParseTree object. 

Parse trees are not generated using an SPPF, which could lead to wrong parse trees [as described on the wikipedia page](https://en.wikipedia.org/wiki/Earley_parser).
My assumption is that the grammars of interest here are non-pathological.

The code for parsing follows the pseudo-code on wikipedia closely.


### The output
I didn't want to bother with sending back complex objects to cython. Instead I return a DFS ordering of the parse tree along.
This is then read from cython to produce the correct python objects.

Serialization code can be found in `SimpleParseTree::serialize`. The format is : 
``` begin_pos length n_children [childrens] ```. 

### The garbage collection
My `SimpleParseTree` dynamic allocation is a complete mess. In order to keep track of allocated objects, I push pointers to a vector that holds all references to be deleted at the end of the parse.

A clean SPPF implementation would be better.

### Testing 
I used `valgrind` on a few examples in `main.cpp` to make sure there are no memory leaks.
Then I ran subst_finder on the entire holophrasm dataset to make sure the parse trees lead to correct substitution inference.


### Possible improvements
- Move all 1-token `class` rules to a specific `#Class_Token#` token to reduce the grammar size. This needs to be dealt with properly when outputting parse trees.
- Stop ridiculous dynamic allocation and EarleyItem hashing. Work with static, pre-allocated memoization tables ?