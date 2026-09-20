# BharatOpt Refinery JSON Model

BharatOpt accepts a generic continuous nonlinear refinery/pooling model as JSON.

## Model structure

Required:

- `variables`: array of named continuous variables with `lower` and `upper`.
- `objective_linear`: optional map of variable name to coefficient.
- `objective_bilinear`: optional array of bilinear terms.
- `constraints`: optional array of named constraints.

A bilinear term has:

```json
{"x": "feed_A", "y": "pool_Diesel", "coefficient": 0.25}
```

This represents:

```
0.25 * feed_A * pool_Diesel
```

A constraint can contain both linear and bilinear terms:

```json
{
  "name": "quality_limit",
  "sense": "<=",
  "rhs": 6.0,
  "linear": {"sulfur_proxy": 1.0},
  "bilinear": [
    {"x": "feed_A", "y": "pool_Diesel", "coefficient": -0.10}
  ]
}
```

Supported senses are `<=`, `>=`, and `=`.

## Solve path

For a JSON model BharatOpt:

1. Parses and structurally validates the model.
2. Searches deterministically for a feasible nonlinear starting point.
3. Runs Sequential Linear Programming (SLP) with a trust region.
4. Solves McCormick convex relaxations.
5. Runs spatial branch-and-bound over bilinear variable bounds.
6. Audits the final solution against the true nonlinear equations.
7. Reports a global optimality certificate only when the remaining global bound is within the configured tolerance.

For maximization, the McCormick relaxation gives an upper bound. A feasible solution is a lower bound. Therefore:

```
global gap = global upper bound - feasible objective
```

The UI reports certification separately from feasibility.

## CLI

```bash
./build/bharatopt \
  --refinery-json examples/refinery_pooling.json \
  --output result.json \
  --mode certified \
  --global-nodes 200 \
  --global-gap 1e-5 \
  --global-time-limit 30
```

## Important scope

This model is a mathematical planning interface, not a process simulator. Plant data should come from validated engineering/process models. The JSON schema can represent pooling, blending, capacity and quality relationships, but it does not claim to reproduce a specific refinery's proprietary unit equations unless those equations are explicitly supplied as model terms.

For a plant deployment, the next data layer should map validated feedstock, unit, yield, product-quality and operating-limit data into this mathematical representation.
