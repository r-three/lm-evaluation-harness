"""
Generates one lm-eval YAML task file per language, plus a group YAML
that lets you run all of them with a single task name: flores200.

Usage:
    python generate_tasks.py
    python lm_eval/tasks/flores/create_configs.py

Then evaluate:
    lm_eval --model hf \
        --model_args pretrained=your-model \
        --tasks flores200 \
        --include_path ./flores200 \
        --device cuda \
        --output_path ./results
"""

import os
from pathlib import Path


LANGS = [
    "arb_Arab",
    "ces_Latn",
    "cmn_Hani",
    "dan_Latn",
    "deu_Latn",
    "ell_Grek",
    "fas_Arab",
    "fra_Latn",
    "eng_Latn",
    "hun_Latn",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "nld_Latn",
    "pol_Latn",
    "por_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "swe_Latn",
    "tur_Latn",
    "vie_Latn",
    "nob_Latn",
    "fin_Latn",
    "ben_Beng",
    "kor_Hang",
]

# lm-eval computes bits-per-byte from loglikelihood_rolling as:
#   bpb = -loglikelihood / (num_bytes * log(2))
# word_perplexity and byte_perplexity are also reported automatically.
TASK_TEMPLATE = """\
task: flores200_{lang}
dataset_path: {repo_id}
dataset_name: {lang}
test_split: devtest
validation_split: dev
# doc_to_text: "{{{{sentence}}}}"
# doc_to_target: ""
doc_to_text: ""
doc_to_target: "{{{{sentence}}}}"
description: "FLORES-200 {lang} — rolling log-likelihood for BPB / perplexity."
output_type: loglikelihood_rolling
# metric_list:
#   - metric: word_perplexity
#     aggregation: perplexity
#     higher_is_better: false
#   - metric: byte_perplexity
#     aggregation: perplexity
#     higher_is_better: false
#   - metric: bits_per_byte
#     aggregation: weighted_perplexity
#     higher_is_better: false
metadata:
  version: 1.0
"""

GROUP_TEMPLATE = """\
group: flores200
task:
{task_list}
# aggregate_metric_list:
#   - metric: byte_perplexity
#     aggregation: mean
#   - metric: bits_per_byte
#     aggregation: mean
metadata:
  version: 1.0
"""


def main(repo_id: str, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)

    task_names = []
    for lang in LANGS:
        task_name = f"flores200_{lang}"
        task_names.append(task_name)
        yaml_path = out_dir / f"{task_name}.yaml"
        with open(yaml_path, "w") as f:
            f.write(TASK_TEMPLATE.format(lang=lang, repo_id=repo_id))
        print(f"  wrote {yaml_path}")

    # Group YAML so `--tasks flores200` runs all languages at once
    task_list_str = "\n".join(f"  - {t}" for t in task_names)
    group_path = out_dir / "_flores200_group.yaml"
    with open(group_path, "w") as f:
        f.write(GROUP_TEMPLATE.format(task_list=task_list_str))
    print(f"  wrote {group_path}")

    print("\nDone. Run with:")
    print("  lm_eval --model hf \\")
    print("      --model_args pretrained=<your-model> \\")
    print("      --tasks flores200 \\")
    print(f"      --include_path {out_dir} \\")
    print("      --device cuda \\")
    print("      --output_path ./results")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    repo = "flexitok/flores-21langs"
    main(repo_id=repo, out_dir=out_dir)
