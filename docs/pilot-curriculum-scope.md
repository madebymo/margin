# Five-goal pilot curriculum scope

This document is the curriculum boundary for the reviewed anonymous pilot. It
narrows broad graph descriptions where the corresponding prerequisite is not
inside the fixed 22-KC hard closure. Item constructors, pedagogy packs, KC
attestations, reviewer packets, goal descriptions, and simulations must all use
these claims. Passing the pilot does not claim mastery outside these domains.

## Shared constraints

- Use real-valued tasks only. Do not require complex numbers.
- Use integer coefficients, inputs, and outputs unless the KC explicitly tests
  symbolic exponent notation. Do not make unprobed fraction arithmetic the
  reason an otherwise correct learner fails.
- Do not require trigonometric, exponential, or logarithmic facts.
- Keep algebraic simplification within the explicitly listed prerequisite KCs.
- Static plots must include reviewed spoken text and equivalent data. A visual
  interpretation item must assess the stated graph construct, not prose recall.
- A contextual story may motivate a task, but it cannot introduce a new
  mathematical prerequisite or alter the answer contract.

## Algebra and functions

| KC | Pilot mastery claim | Excluded from this release |
|---|---|---|
| `kc.alg.exponent_rules` | Simplify products, quotients, powers, zero exponents, and negative **integer** exponents for one symbolic base. | Fractional exponents, radical conversion, numeric fraction arithmetic, domain analysis. |
| `kc.alg.polynomial_ops` | Add, subtract, expand, and collect polynomials of degree at most three with small integer coefficients. | Polynomial division, rational expressions, multivariable identities. |
| `kc.alg.factoring` | Factor a greatest common monomial, a difference of squares, or a monic quadratic with integer roots. | Irreducible/complex factors, nonmonic rational-root search, completing the square. |
| `kc.alg.solve_linear` | Solve one-variable linear equations deliberately parameterized to have integer solutions. | Fraction arithmetic as the target, inequalities, absolute values, systems. |
| `kc.alg.solve_quadratic` | Solve monic, factorable quadratic equations with distinct or repeated integer roots and report a finite set. | Quadratic formula, completing the square, irrational or complex roots. |
| `kc.fun.function_notation` | Evaluate and interpret polynomial or affine functions at integer inputs or simple symbolic expressions. | Piecewise ambiguity, inverse functions, nonalgebraic function facts. |
| `kc.fun.graph_reading` | Read integer-valued points, intercepts, intervals of increase/decrease, and constant slopes from reviewed piecewise-linear plots and equivalent tables. | Limit behavior, discontinuity classification, nonlinear numerical estimation. |
| `kc.fun.composition` | Build, evaluate, and identify inner/outer structure for affine and polynomial compositions. | Trigonometric/exponential/logarithmic identities, inverse composition. |

## Differentiation

| KC | Pilot mastery claim | Excluded from this release |
|---|---|---|
| `kc.der.power_rule` | Differentiate integer powers, including zero and negative integer exponents, using symbolic exponent notation. | Radicals, fractional exponents, domain restrictions. |
| `kc.der.sum_constant_rules` | Differentiate integer-coefficient polynomial sums, differences, and constant multiples term by term. | Trigonometric, exponential, and logarithmic derivative tables. |
| `kc.der.product_quotient` | Apply product and quotient rules to reviewed point-value data or polynomial factors whose arithmetic yields an integer/symbolic result without unrelated simplification. | Trig/exp/log derivatives, quotient-domain analysis, fraction arithmetic as a hidden gate. |
| `kc.der.chain_rule` | Differentiate an integer power of an affine or polynomial inner function and identify the inner-derivative factor. | Trig/exp/log outer functions, implicit differentiation, nested products requiring unprobed rules. |
| `kc.der.differentials` | Convert a polynomial substitution `u=g(x)` into `du=g'(x) dx` and identify an exact matching factor. | Error approximation, separable equations, multivariable differentials. |

## Integration and accumulation

| KC | Pilot mastery claim | Excluded from this release |
|---|---|---|
| `kc.int.area_under_curve` | Accumulate nonnegative rectangular and triangular regions with integer dimensions from reviewed plots/tables. | General signed net area, curved-region approximation, physical unit conversion. |
| `kc.int.riemann_sums` | Compute left, right, or midpoint sums from a table on equal-width partitions with integer widths and values. | Limit proofs, sigma-algebra manipulation, fractional-width arithmetic as a hidden gate. |
| `kc.int.definite_integral` | Interpret bounds, orientation, and signed accumulation, and connect a reviewed finite accumulation to definite-integral notation. | Improper integrals, numerical quadrature error bounds, discontinuous integrability proofs. |
| `kc.int.antiderivatives` | Reverse the integer power rule for polynomial terms whose coefficients produce simple exact antiderivatives. | Trig/exp/log tables, partial fractions, integration techniques. |
| `kc.int.indefinite_notation` | Express a polynomial antiderivative family with exactly one arbitrary constant and distinguish it from a definite value. | Initial-value problems, differential equations, multiple constants. |
| `kc.int.ftc` | Evaluate polynomial definite integrals by applying a supplied or readily derived polynomial antiderivative at integer bounds. | Variable-bound derivative forms, improper integrals, trig/exp/log evaluation. |
| `kc.int.integral_basic_rules` | Integrate polynomial sums term by term using the reverse integer power rule and constant multiples. | Trigonometric, exponential, logarithmic, rational-function, or radical integration tables. |
| `kc.int.recognizing_composite` | Identify an affine/polynomial inner function and an exact constant multiple of its derivative in a power-composite integrand. | Algebraic completion tricks, trig identities, partial matching that requires a new technique. |
| `kc.int.u_substitution` | Evaluate indefinite integrals of reviewed polynomial-power composites by substituting the inner polynomial and converting the full differential. | Definite-bound conversion, trig/exp/log substitution, integration by parts, partial fractions. |

## Review enforcement

For every KC, the coverage attestation must repeat the exact mastery claim and
list its task constructors. Publication fails if an item constructor declares a
domain outside the claim, if an excluded function appears in a prompt/answer,
or if the first two diagnostic families cover only one narrow subconstruct.

Any desired expansion requires all of the following before authoring: graph
review, explicit hard-prerequisite modeling, a new graph version, updated
coverage attestations, and new release coordinates. It must not be introduced
by changing only prose or task parameters.
