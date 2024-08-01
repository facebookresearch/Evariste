Equation
========
The Equation Environment is a toy formal proving environment that is used to manipulate simple mathematical expressions comprising equalities and inequalities between real numbers, or subset (subtype) of real numbers, elementary arithmetic objects (integers, divisibility, gcd ...)


Graph.py 
--------
This file :

* contains the building blocks of the environment, the object `Node` which the basis of each expression. Each expression, integer, variable, etc. is a `Node`. For instance `x + y` is the `Node` with value `+` and two children that are the `Node` `x` and `y`.

* `Node` objects can easily be manipulated using Python operators. Using Python operators will create new nodes. 

Example::

   x = VNode('x0')
   y = VNode('x1')
   z = VNode('x2')
   PI = PNode("PI")

   ZERO = INode(0)
   ONE = INode(1)

   print(x + y)
   print((x + y).value)
   print((x + y).children)
   print(((x+y)**0.5))
   print(((x+y)**0.5).children)
   print((x + y).exp() + 3 - 1)
   print((PI / 2).cos() == ZERO)
   print(y > (x + 1).ln())

will output::

   (x0 + x1)
   add
   [x0, x1]
   sqrt(x0 + x1)
   [(x0 + x1)]
   ((exp(x0 + x1) + 3) - 1)
   (cos(PI / 2) == 0)
   (ln(x0 + 1) < x1)


Node
~~~~
The methods of the `Node` class are the following

----

.. autoclass:: evariste.envs.eq.graph.Node
   :members: Node

----

Expressions should not be compared with the `==` operator, as it is overloaded (and so are `!=`, `<=`, `<`), but with the `.eq` (or `.ne` for non equal) functions.


Rules.py
--------
This file contains one of the main components of the environment, i.e. the rules that are the building blocks of tactics.

There are two types of rules:

* transformation rules (`TRule`)

* assertion rules (`ARule`)

Tranformation rules
~~~~~~~~~~~~~~~~~~~
The transformation rules express the fact that two quantities are mathematically the same and could be replaced one by the other, sometimes provided some assumptions. For example, in any expression `A+B` could be replaced by `B+A` without any assumption. They are composed of three arguments: a source expression, a target expression, and a set of hypotheses for the source to be equal to the target. Some examples::
   
   A = VNode("A")
   B = VNode("B")
   C = VNode("C")

   rule1 = TRule(A + B, B + A)
   rule2 = TRule(ZERO, A - A)
   rule3 = TRule(A, exp(ln(A)), hyps=[A > 0])

The rules can be applied to expressions by providing the direction in which the rule is applied, and the position where we apply it.

Applying a rule results in a dictionary that contains the new equation, and the matched variables content::

   rule1.apply(x + y ** 2, fwd=True, prefix_pos=0)

The `prefix_pos` parameter allows to apply the rule to different subtrees of the input expression::
   
   print(rule1.apply(((((x + 1) + y) + 2) + 3), True, 0)["eq"])
   print(rule1.apply(((((x + 1) + y) + 2) + 3), True, 1)["eq"])
   print(rule1.apply(((((x + 1) + y) + 2) + 3), True, 2)["eq"])
   print(rule1.apply(((((x + 1) + y) + 2) + 3), True, 3)["eq"])

will yield::
   
   (3 + (((x0 + 1) + x1) + 2))
   ((2 + ((x0 + 1) + x1)) + 3)
   (((x1 + (x0 + 1)) + 2) + 3)
   ((((1 + x0) + x1) + 2) + 3)

The `to_fill` value corresponds to the set of expressions that remain to be generated.

Sometimes, there is nothing to generate, but sometimes, some variables appear in the target expression, and not in the source one, so we need to generate them.

For instance, if we transform `0` in `A - A`, the value of `A`is not imposed by the source expression and we need to generate it::

   rule2.apply(ZERO, fwd=True, prefix_pos=0)

will yield::

   {'eq': (A - A), 'match': {}, 'to_fill': {'A'}, 'hyps': []}

----

.. autoclass:: evariste.envs.eq.rules.TRule
   :members:   

----

Assertion rules
~~~~~~~~~~~~~~~
The assertion rules are more similar to what we would observe in Metamath. They express the fact that an expression is true provided some assumptions.

They contain an expression, and a list of hypotheses (possibly empty) for the expression to hold.

For instance::
   
   rule_a1 = ARule(exp(A) > 0, hyps=[])
   rule_a2 = ARule(exp(A) < exp(B), hyps=[A < B])
   rule_a3 = ARule(A <= B, hyps=[A <= C, C <= B])

Unlike transformation rules, assertion rules do not require any direction (forward or backward) or prefix position, as they act on whole expressions::
   
   print(rule_a1.apply(exp(x - y ** 2) > 0))
   print(rule_a2.apply(exp(x + 3) < exp(-y)))   

will yield::
   
   {'match': {'A': (x0 - (x1 ** 2))}, 'hyps': [], 'to_fill': set()}
   {'match': {'A': (x0 + 3), 'B': (-(x1))}, 'hyps': [((x0 + 3) < (-(x1)))], 'to_fill': set()}

----

.. autoclass:: evariste.envs.eq.rules.TRule
   :members:   

----


eval_assert
~~~~~~~~~~~
This is an important function, that takes as input an equation, and a list of assertion rules, and returns:

* `True` if the expression is surely `True` (e.g. `exp(x + y) > 0`, or `x ** 2 >= 0`)

* `False` if the expression is surely `False` (e.g. `exp(x + y) < 0`, or `x ** 2 < 0`)

* `None` if the truth of the expression cannot be obviously inferred given the list of provided assertion rules (for instance if it depends of the variables in a non obvious way, e.g. `x > 0`)

To determine that `exp(x + y) > 0` is `True`, the `eval_assert` will leverage the rule `ARule(exp(A) > 0, hyps=[])` which does not have any hypothesis, and matches the  
equation for `A = x + y`. `eval_assert` will know that `x ** 2 < 0` is `False` because its negated expression (`x ** 2 >= 0`) matches the rule `Arule(A ** 2 >= 0)`. 

On `exp(x) < exp(y)`, `eval_assert` will return `None`, as the assertion rule `ARule(exp(A) < exp(B), hyps=[A < B])` will require some hypotheses to be verified for the 
expression to be true (i.e. `x < y`). Overall, `eval_assert` only uses the assertion rules that have no hypotheses. 

The function `eval_assert` is very useful as it allows us to avoid obviously `False`expressions, but it is not a prover.  For instance even if `ARule(A ** 2 >= 0)` is provided, `eval_assert` will still return `None` for `x ** 2 + y ** 2 >= 0` because this requires at least a one-step proof. Similarly it will also return `None` on a simple expression like `exp(ZERO * x) == ONE` because this also requires a one-step proof.

If the expression does not have any variables involved (so is constant), `eval_assert` will numerically evaluate the left and the right hand side of the expression, and return
`True` or `False` by comparing the results. If the expression is invalid, because the left hand side evaluates to None (e.g. `ln(-1)`) `eval_assert` will return `None`.

.. autoclass:: evariste.envs.eq.rules
   :members: eval_assert,  


Generation.py
-------------

This file contains `EquationGraphGenerator`, the main generator class. The generator can generate random statements, the hypotheses that need to be verified
for these statements to hold, and the proof (i.e. succession of rules/tactics) that lead to these statements.

`EquationGraphGenerator` implements two different random generators:

* random walk (uses transformation rules)

* random graph (uses both transformation and assertion rules)

Let's build an environment, and a generator::
   
   args = ConfStore["eq_env_basic"]
   args.binary_ops = ["add", "mul"]
   args.unary_ops = ["neg", "inv"]
   # args.unary_ops = ["neg", "inv", "exp", "ln", "pow2", "sqrt", "cos", "sin", "tan", "cosh", "sinh", "tanh"]
   args.n_vars = 5
   args.seed = 0
   args.check_and_mutate_args()
   env = EquationEnv.build(args)

   # load rules
   # NOTE: the difference between rules_t_e and rules_t_c is simply that rules_t_c are applied
   # to compositions (i.e. A == B or A < B) while rules_t_e can be applied to any expressions.
   # rules_t_c are actually replacing the rules_a for the random_walk, which only uses transformation rules.

   rules = [
      rule
      for rule in RULES_T_E + RULES_T_C + RULES_A
      if rule.get_unary_ops().issubset(env.unary_ops)
      and rule.get_binary_ops().issubset(env.binary_ops)
   ]

   # variables
   x = VNode("x0")
   y = VNode("x1")
   z = VNode("x2")
   t = VNode("x3")
   node_vars = NodeSet(RULE_VARS)

   # graph generator
   egg = EquationGraphGenerator(
      env=env,
      rules=rules,
      hyp_max_ops=3,
      tf_prob=0.5,
      bias_nodes=0,
      bias_rules=1,
   )

----

.. autoclass:: evariste.envs.eq.generation.GraphNode
   :members: 

.. autoclass:: evariste.envs.eq.generation.EquationGraphGenerator
   :members: find_all_matches, add_node, find_all_matches, create_transformation_node, create_assertion_node


----



Random Walk
~~~~~~~~~~~
The random walk generator consists in generating a random expression `init_eq`, and randomly applying transformation rules to modify this expression. 
Sometimes, these tactics require some hypotheses. For instance, transforming `ln(x * y)` into `ln(x) + ln(y)` requires the expressions `x > 0` and `y > 0` 
to be verified. The generator will potentially add these expressions to the list of hypotheses required for the walk to be valid with a given probability `prob_add_hyp` (see below).

In order to not add an unreasonnable number of hypotheses to the random walk, the random walk takes as input an argument `max_created_hyps` that bounds 
the number of hypotheses. Once `max_created_hyps` hypotheses have been created, the generator will only be able to apply rules that do not involve new hypotheses. 
Of course, it is totally possible that the generator rejects an hypothesis because it is not in the list, although it would be true given the current hypotheses  
(for instance, the generator will not know that `x + 1 > 0` if `x > 0` is an hypothesis). To prevent the generator from immediately creating too many hypotheses, 
the generator takes as input a probability `prob_add_hyp` that defines the probability of accepting a new hypothesis. A low probability will encourage the generator 
to first use rules that do not require hypotheses, and to add them later during the generation, while a high probability will quickly create `max_created_hyps` hypotheses.

Here's a code example::

   egg.env.rng = np.random.RandomState(5)

   walk = egg.random_walk(
      bidirectional=False,
      n_steps=5,
      n_init_ops=3,
      max_created_hyps=2,
      prob_add_hyp=0.5,
   )

   print("Initial equation:")
   print(f'    {walk["init_eq"]}')

   print("\nSteps:")
   for step in walk["steps"]:
      print(f'    {step["eq"]}')

   print("\nHypotheses:")
   for hyp in walk["hyps"]:
      print(f'    {hyp}')

This will yield::

   Initial equation:
    ((-(((-6) * x0) + (-7))) < ((-6) * (-9)))

   Steps:
      ((-(((-6) * x0) + (-7))) < ((-6) * (-9)))
      (((-(((-6) * x0) + (-7))) * 1) < ((-6) * (-9)))
      (((-(((-6) * x0) + (-7))) * 1) < ((-9) * (-6)))
      (((-(-(((-6) * x0) + (-7)))) * (-(1))) < ((-9) * (-6)))
      (((-(-(((-6) * x0) + (-7)))) * (-(1))) < ((-6) * (-9)))
      (((-(-((x4 * x3) * x3))) * (-(1))) < ((-6) * (-9)))

   Hypotheses:
      (((x4 * x3) * x3) == (((-6) * x0) + (-7)))


----

.. autoclass:: evariste.envs.eq.generation.EquationGraphGenerator
   :members: random_walk

----

Graph generation
~~~~~~~~~~~~~~~~

The generator builds a graph of nodes, of type `GraphNode`. The idea is that each node of the graph is a `Node` expression that can be deduced from the others using the rules provided.

The class `GraphNode` is very simple, and contains three main attributes:

* `node`, the current node content (type `Node`)

* `hyps`, the list of children that lead to that goal (type `List[GraphNode]`)

* `ntype`, a string that can take 4 values:

  * `transform` if the node was created by a transformation rule

  * `assert` if it was created by an assertion rule

  * `hyp` if this is an initial hypothesis of the graph

  * `true` if this is a node trivially true (as indicated by the `eval_assert` function)

For a `GraphNode` of type `transform`, generated by a `TRule`, the node also contains extra attributes that describe the applied rule:

* `rule` the applied transformation rule

* `fwd` the direction in which it was applied (forward or backward) to produce the current node

* `prefix_pos` the prefix position where it was applied

* `substs` the substitutions involved

For a `GraphNode` of type `assert`, generated by a `ARule`, we only need to store:

* `rule` the applied assertion rule

* `substs` the substitutions involved

The generator is called by the function `generate_graph` with takes in input a number of nodes to generate in the graph `nr_nodes`, a maximal number of trials `max_trials`, and a number of random initial hypotheses `n_init_hyps`.  This generator outputs a list of `GraphNode`and a list of initial hypotheses (type `Node`).

In the generation the function will first select a rule and try to apply it somewhere on the graph. If this rule cannot be applied on any node of the graph or if the selected node does not have the right assumptions for the rule to be applied no node is added and a new rule is selected. This explains why the number of trials is likely to be different from the number of nodes to generate.

For example::

   nodes, init_hyps = egg.generate_graph(n_nodes=10, max_trials=30, n_init_hyps=2
        )
   print('== Initial assumptions ==')
   for hyp in init_hyps:
      print(hyp)

   print('\n== Nodes generated ==')
   for node in nodes:
      print(node.node) #Recall that node has type GraphNode, hence node.node is a Node

will output::

   == Initial assumptions ==
   (((x0 * (-(4))) + x3) == (x0 ** -1))
   ((-(x0 + x3)) == ((1 * (-(x1))) + x2))

   == Nodes generated ==
   (((x0 * (-(4))) + x3) == (x0 ** -1))
   ((-(x0 + x3)) == ((1 * (-(x1))) + x2))
   (((1 * (-(x1))) + x2) == ((1 * (-(x1))) + x2))
   ((((1 * (-(x1))) + x2) * ((-2) ** -1)) == (((1 * (-(x1))) + x2) * ((-2) ** -1)))
   ((1 * (-(x1))) == (1 * (-(x1))))
   (((1 ** -1) * (-(x1))) == (1 * (-(x1))))
   (((x0 * (-4)) + x3) == (x0 ** -1))
   ((((-(x0)) * (-(-(4)))) + x3) == (x0 ** -1))
   (((((1 * (-(x1))) + x2) * ((-2) ** -1)) * (((-1) + 6) * x0)) == ((((1 * (-(x1))) + x2) * ((-2) ** -1)) * (((-1) + 6) * x0)))
   (((1 ** -1) * (-(x1))) <= (1 * (-(x1))))

For each node it is possible to extract all the steps used to derive them from the initial assumptions, just like in the random walk::

   def print_node(node: GraphNode, depth=0):
    prefix = "\t" * depth
    ntype = node.ntype.upper()
    print(f"{prefix}{ntype} | {node.node}")
    if node._hyps is not None:
        for hyp in node.hyps:
            print_node(hyp, depth=depth + 1)
            
   def _get_hyps(node: GraphNode, res: NodeSet):
      if node.ntype == "hyp":
         res.add(node.node)
      if node._hyps is not None:
         for hyp in node.hyps:
               _get_hyps(hyp, res)

   nodes, init_hyps = egg.generate_graph(n_nodes=50, max_trials=300, n_init_hyps=2
         )
   node = nodes[len(nodes)-1]
   print("== final node ==")
   print(node.node)
   print("\n== derived from initial hypothesis ==")
   hyps = NodeSet()
   _get_hyps(node,hyps)
   if len(hyps) == 0:
      print('No hyps')
   else:
      for hyp in hyps:
         print(hyp)
   print("\n== Steps ==")
   print_node(node)

will yield::

   == final node ==
   (((-1) + ((x2 * (-2)) * x3)) <= ((-1) + ((x2 * ((-2) + 0)) * x3)))

   == derived from initial hypothesis ==
   No hyps

   == Steps ==
   TRANSFORM | (((-1) + ((x2 * (-2)) * x3)) <= ((-1) + ((x2 * ((-2) + 0)) * x3)))
      TRUE | (((-1) + ((x2 * (-2)) * x3)) <= ((-1) + ((x2 * (-2)) * x3)))

----

.. autoclass:: evariste.envs.eq.generation.EquationGraphGenerator
   :members: generate_graph

----
