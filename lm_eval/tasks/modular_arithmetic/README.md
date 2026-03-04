# Modular Arithmetic

### Dataset

Homepage: https://huggingface.co/datasets/flexitok/mod-arithmetic

A synthetic dataset for evaluating language model ability to compute `a mod b` for various moduli.
Each example presents the problem as a completion task: given `"{a} mod {b} ="`, the model must
generate the correct integer remainder.

- `a` is drawn uniformly from [0, 9999]
- Moduli: 2, 3, 4, 5, 8, 10, 16, 25, 32, 100, 125, 1000
- Splits: 108,000 train / 12,000 test (90/10 split, seed 42)

### Tokenizer Hypothesis

For `a mod b` where `b = 2^k × 5^j` (no other prime factors), only the rightmost `max(k, j)`
digits of `a` determine the answer, because `10^max(k,j) ≡ 0 (mod b)`. A tokenizer that groups
digits right-to-left in chunks of that size exposes the relevant information as a single token.

For all other moduli (e.g. 3, 7, 11, …) the digit structure is more complex and is labelled
`complex`.

| modulus (b) | suitable_tokenizer | problem_type    |
|------------:|:------------------:|:----------------|
|           2 | digit_1_rtl        | mod_power_of_2  |
|           3 | complex            | mod_prime       |
|           4 | digit_2_rtl        | mod_power_of_2  |
|           5 | digit_1_rtl        | mod_power_of_5  |
|           8 | digit_3_rtl        | mod_power_of_2  |
|          10 | digit_1_rtl        | mod_power_of_10 |
|          16 | digit_4_rtl        | mod_power_of_2  |
|          25 | digit_2_rtl        | mod_power_of_5  |
|          32 | digit_5_rtl        | mod_power_of_2  |
|         100 | digit_2_rtl        | mod_power_of_10 |
|         125 | digit_3_rtl        | mod_power_of_5  |
|        1000 | digit_3_rtl        | mod_power_of_10 |

### Dataset Fields

| field              | type | description                           |
|:-------------------|:----:|:--------------------------------------|
| a                  | int  | dividend                              |
| b                  | int  | modulus                               |
| answer             | int  | a mod b                               |
| text               | str  | "{a} mod {b}"                         |
| problem_type       | str  | classification of b                   |
| suitable_tokenizer | str  | hypothesised best tokenizer type      |

### Task Format

- **Input** (`doc_to_text`): `"{{text}} ="` → e.g. `"1234 mod 7 ="`
- **Target** (`doc_to_target`): `"{{answer}}"` → e.g. `"6"`
- **Output type**: `generate_until` (stops at newline)
- **Metric**: `exact_match` on the first integer extracted from the generated output

### Groups, Tags, and Tasks

#### Tags

* `modular_arithmetic`: parent tag for all modular arithmetic tasks

#### Tasks

Child tasks include `base.yaml` and specify a `task` name and (if applicable) a `dataset_name`
for a particular modulus subset:

* `modular_arithmetic_mod2`
* `modular_arithmetic_mod3`
* `modular_arithmetic_mod4`
* `modular_arithmetic_mod5`
* `modular_arithmetic_mod8`
* `modular_arithmetic_mod10`
* `modular_arithmetic_mod16`
* `modular_arithmetic_mod25`
* `modular_arithmetic_mod32`
* `modular_arithmetic_mod100`
* `modular_arithmetic_mod125`
* `modular_arithmetic_mod1000`

### Generation

```bash
python -m flexitok.simplified.create_numeric_synthetic_data \
    hf.hf_repo_id=flexitok/mod-arithmetic hf.publish_to_hf=true \
    a_min=0 a_max=9999 \
    moduli="[2, 3, 4, 5, 8, 10, 16, 25, 32, 100, 125, 1000]" \
    seed=42 train_ratio=0.9
```

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you
        checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* version 1.0: initial release
