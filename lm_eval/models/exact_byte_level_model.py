import json
import os
from typing import List, Optional

import torch

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from .exact_byte_level.byte_ensemble_model import ByteEnsemble


# hf download google/gemma-2-2b --include tokenizer.model --local-dir  .venv/lib/python3.10/site-packages/lm_eval/models/exact_byte_level
# Register the model adapter with lm-eval-harness
@register_model("byte_level_llm")
class ByteLevelAdapter(LM):
    
    # Define the required constructor arguments
    def __init__(
        self,
        model_paths: List[str],  # Can be a single path
        tokenizer_paths: Optional[List[str]] = None,      # For ByteEnsemble, a list of paths is needed
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        # import code; code.interact(local=locals()|globals())
        if isinstance(model_paths, str):
            model_paths = json.loads(model_paths)
        if isinstance(tokenizer_paths, str):
            tokenizer_paths = json.loads(tokenizer_paths)
        # Determine the paths to load the model(s)
        if tokenizer_paths is None:
            tokenizer_paths = model_paths
            
        print(f"Loading ByteEnsemble with paths: {model_paths} on device: {device}")
        
        # Initialize your custom model/ensemble
        # NOTE: ByteEnsemble handles device mapping internally based on the number of paths
        self.model = ByteEnsemble(paths=model_paths, tokenizer_paths=tokenizer_paths, device=device)
        self.device = self.model.models[0].device # Inherit device from the first model

        # Required lm-eval attributes
        self.tokenizer = self.model.models[0].t_wrapper # Use your ByteTokenizer wrapper
        self.vocab_size = self.tokenizer.vocab_size
        self.eot_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def loglikelihood(self, requests):
        res = []
        
        # Requests is a list of lm_eval Request objects (or Instance objects)
        for req in requests: # <-- Iterate over the request object itself
            # Extract context and continuation strings directly from the attributes
            # The structure depends on the specific lm_eval version, but the attributes 
            # should generally be accessible via the request's args property.
            context, continuation = req.args 

            # 1. Ensure context/continuation are strings (they should be if loaded correctly)
            # The original decode/check logic is often safer if lm-eval doesn't guarantee strings
            
            # The next lines remain the same, using the extracted strings
            if isinstance(context, list): 
                context = self.tokenizer.decode(context)
            if isinstance(continuation, list):
                continuation = self.tokenizer.decode(continuation)

            # 2. Call your custom byte-level log-likelihood function
            log_prob = self.model.compute_byte_loglikelihood(
                prompt=context,
                completion=continuation
            )
            
            # 3. Store the result
            res.append((log_prob, False))

        return res
    # @torch.no_grad()
    # def loglikelihood(self, requests):
    #     res = []
        
    #     for context, continuation in requests:
    #         # 1. Decode context/continuation from lm-eval's token IDs (optional step, but safe)
    #         # Since lm-eval often gives context/continuation as strings, we ensure they are strings
    #         if isinstance(context, list): # lm-eval sometimes passes token IDs
    #             context = self.tokenizer.decode(context)
    #         if isinstance(continuation, list):
    #             continuation = self.tokenizer.decode(continuation)

    #         # 2. Call your custom byte-level log-likelihood function
    #         # This function calculates: log P(continuation | context)
    #         log_prob = self.model.compute_byte_loglikelihood(
    #             prompt=context,
    #             completion=continuation
    #         )
            
    #         # 3. Store the result
    #         # lm-eval expects (log_likelihood, is_greedy)
    #         # Since we are calculating the exact log P, is_greedy is False/None.
    #         res.append((log_prob, False))

    #     return res

    # 3. Implement required dummy methods (for byte-level, these are simple)
    def loglikelihood_rolling(self, requests):
        # For byte-level, rolling likelihood is complex. 
        # For simple completion tasks, we can defer to loglikelihood.
        raise NotImplementedError("")

    def generate_until(self, requests):
        # This is for open-ended generation tasks, not required for completion-style evaluation.
        raise NotImplementedError("")