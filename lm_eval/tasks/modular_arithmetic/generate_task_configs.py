from pathlib import Path

import yaml


tasks = ["mod_2", "mod_3", "mod_4", "mod_5", "mod_7", "mod_8", "mod_10", "mod_16", "mod_25", "mod_32", "mod_100", "mod_125", "mod_1000"]

# python lm_eval/tasks/modular_arithmetic/generate_task_configs.py

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    for task in tasks:
        with open(base_dir / f"{task}.yaml", "w") as f:
            cfg_dict = {"task": task, "include": "base.yaml", "dataset_name": task}
            yaml.dump(cfg_dict, f)
    
    group_cfg_dict = {"group": "modular_arithmetic", "task": tasks, "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}], "metadata": {"version": 1}}
    with open(base_dir / "_modular_arithmetic.yaml", "w") as f:
        yaml.dump(group_cfg_dict, f)



# task:
#   - mmlu_stem
#   - mmlu_other
#   - mmlu_social_sciences
#   - mmlu_humanities
# aggregate_metric_list:
#   - metric: acc
#     weight_by_size: True
# metadata:
#   version: 2