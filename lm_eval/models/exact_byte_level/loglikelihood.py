# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch  # Import torch for log operations and device handling
from sklearn.covariance import log_likelihood

sys.path.append("./src")

# Assuming ByteEnsemble and BytePredLLM with the new method are accessible
import argparse

from src.byte_ensemble_model import ByteEnsemble

# python loglikelihood.py --model_paths=r-three/supertoken_models-llama_google-gemma-2-2b --tokenizer_paths=google/gemma-2-2b  --temp=0 --device=cuda
# python loglikelihood.py --model_paths=01-ai/Yi-1.5-6B --tokenizer_paths=01-ai/Yi-1.5-6B  --temp=0 --device=cuda

parser = argparse.ArgumentParser(description="Input a list of paths for multiple-choice evaluation.")
parser.add_argument("--model_paths", nargs="+", help="List of paths to model checkpoints.", required=True)
parser.add_argument("--tokenizer_paths", nargs="+", help="List of paths to model checkpoints.", default=None)
parser.add_argument("--prompt", type=str, help="Prompt/Question for the evaluation.", default="What is the capital of France?")
parser.add_argument("--device", type=str, help="device", default='cuda')
parser.add_argument("--temp", type=float, help="Temperature (not used for likelihood evaluation)", default=0.6)
parser.add_argument(
    "--max_new_bytes", type=int, help="max bytes to generate (not used for likelihood evaluation)", default=500
)

# --- Minimal Multiple-Choice Evaluation Script ---
def main():
    args = parser.parse_args()
    if args.tokenizer_paths is None:
        args.tokenizer_paths = args.model_paths

    device = args.device
    
    # 1. Load Model/Ensemble
    print(f"Loading ensemble models onto device(s): {device}")
    # Load model(s). If using ByteEnsemble, the models will be loaded onto different GPUs
    # or the specified single device if 'cpu' or single 'cuda:X'.
    try:
        mixture = ByteEnsemble(args.model_paths, args.tokenizer_paths, device=device)
    except Exception as e:
        print(f"Error loading models: {e}")
        # Exit if model loading fails
        return

    # 2. Define the Evaluation Task
    # For a multiple-choice question format (log-likelihood evaluation)
    prompt = args.prompt
    
    # Example options for the question "What is the capital of France?"
    # In a real benchmark, these would be loaded from a dataset file.
    options = {
        "A": " Paris",
        "B": " Rome",
        "C": " Berlin",
        "D": " Madrid"
    }

    print("-" * 50)
    print(f"Evaluation Prompt: **{prompt}**")
    print("-" * 50)

    # 3. Compute Log-Likelihoods for all Options
    log_likelihoods = {}
    best_log_likelihood = -float('inf')
    predicted_option = None

    for key, completion in options.items():
        # NOTE: The completion often includes a leading space or delimiter for the model to continue the sentence.
        print(f"  Calculating log-likelihood for Option {key}: '{completion.strip()}'...")
        
        # Calculate P(completion | prompt) using the new method
        # This will internally sum the log P(byte_i | context) for all bytes in the completion.
        log_prob = mixture.compute_byte_loglikelihood(prompt, completion)
        
        log_likelihoods[key] = log_prob
        
        print(f"    -> Log-Likelihood: {log_prob:.4f}")

        # 4. Determine the Best Option (Highest Log-Likelihood)
        if log_prob > best_log_likelihood:
            best_log_likelihood = log_prob
            predicted_option = key

    print("-" * 50)
    print("### 🏆 Final Prediction ###")
    print(f"| Option | Log-Likelihood |")
    print(f"| :----: | :-------------: |")
    for key, log_prob in log_likelihoods.items():
        # Highlight the predicted option
        marker = "<- PREDICTED" if key == predicted_option else ""
        print(f"| {key} | {log_prob:.4f} {marker} |")
        
    print(f"\nModel Prediction: **Option {predicted_option}** (Log-Likelihood: {best_log_likelihood:.4f})")
    print("-" * 50)


if __name__ == "__main__":
    main()