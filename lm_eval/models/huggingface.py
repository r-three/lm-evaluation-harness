from __future__ import annotations

import copy
import logging
import os
from collections.abc import Iterator, Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jinja2
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from accelerate.utils import get_max_memory
from huggingface_hub import HfApi
from packaging import version
from packaging.version import parse as vparse
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    postprocess_generated_text,
    stop_sequences_criteria,
)


# Import your custom tokenizer classes - adjust the import path as needed
try:
    from lm_eval.models.tokenizers_module import (
        MistralTokenizer,
        TikTokenTokenizer,
        Tokenizer,
        TokenMonsterTokenizer,
    )

    CUSTOM_TOKENIZERS_AVAILABLE = True
except ImportError:
    CUSTOM_TOKENIZERS_AVAILABLE = False

if TYPE_CHECKING:
    from transformers.quantizers.auto import AutoQuantizationConfig

    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)
TOKENIZER_INFINITY = 1000000000000000019884624838656

if not CUSTOM_TOKENIZERS_AVAILABLE:
    eval_logger.warning(
        "Custom tokenizers not available. Only HuggingFace tokenizers will be supported."
    )


@register_model("hf-auto", "hf", "huggingface")
class HFLM(TemplateLM):
    """An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Supports data-parallel multi-GPU with HF Accelerate.
    """

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: str | None = "main",
        subfolder: str = "",
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None = None,
        truncation: bool | None = False,
        logits_cache: bool = True,
        max_length: int | None = None,
        device: str | None = "cuda",
        dtype: str | torch.dtype | None = "auto",
        softmax_dtype: str | torch.dtype | None = None,
        mixed_precision_dtype: str | torch.dtype | None = None,
        batch_size: int | str | None = 1,
        max_batch_size: int | None = 64,
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        add_bos_token: bool | None = False,
        prefix_token_id: int | None = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: bool | None = False,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | os.PathLike | None = "./offload",
        # PEFT, delta weights and quantization options
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        # end token for thinking, either the string or int token id.
        # splits to get response after this token (if provided).
        think_end_token: str | int | None = None,
        enable_thinking: bool | None = None,
        chat_template_args: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # optionally: take in an already-initialized transformers.PreTrainedModel
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, (
                "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            )
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()

            # using one process with no model parallelism
            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:  # Parallelism managed by accelerate
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )

            revision = str(revision)  # cast to string if not already one

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
                subfolder=subfolder,
            )

            # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )

        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )

        if (
            quantization_config := getattr(self.config, "quantization_config", None)
        ) is not None and isinstance(quantization_config, dict):
            from transformers.quantizers import AutoQuantizationConfig

            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        # if we passed `pretrained` as a string, initialize our model now
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                quantization_config=quantization_config,
                subfolder=subfolder,
                **kwargs,
            )

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.think_end_token = (
            int(think_end_token)
            if (isinstance(think_end_token, str) and think_end_token.isdigit())
            else think_end_token
        )
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)
        self.chat_template_args = (
            chat_template_args or {} | dict(enable_thinking=enable_thinking)
            if enable_thinking is not None
            else {}
        )

        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        self.softmax_dtype = (
            get_dtype(softmax_dtype) if softmax_dtype is not None else None
        )
        self.mixed_precision_dtype = (
            get_dtype(mixed_precision_dtype)
            if mixed_precision_dtype is not None
            else None
        )

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if isinstance(pretrained, str):
            if (gpus >= 1 or str(self.device) == "mps") and not (
                parallelize or autogptq or hasattr(self, "accelerator")
            ):
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                # place model onto device requested manually,
                # if not using HF Accelerate or device_map
                # or any other option that preloads model onto device
                try:
                    self.model.to(self.device)
                except ValueError:
                    eval_logger.debug(
                        "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                    )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

    def _should_use_custom_tokenizer(self, tokenizer_name: str) -> bool:
        """Determine if we should use a custom tokenizer based on tokenizer name."""

        if not CUSTOM_TOKENIZERS_AVAILABLE:
            return False

        # Check for explicit custom tokenizer specifications from your TOKENIZER_NAMES
        custom_tokenizer_patterns = [
            "tokenmonster/*",
            "tiktoken/gpt-4o",
            "mistralai/tekken",
        ]
        import re

        return any(
            re.match(pattern, tokenizer_name) for pattern in custom_tokenizer_patterns
        )

    def _wrap_custom_tokenizer(self, custom_tokenizer) -> object:
        """Create a wrapper that makes custom tokenizers compatible with HF interface."""

        class CustomTokenizerWrapper:
            def __init__(self, custom_tok):
                self.custom_tokenizer = custom_tok
                self._vocab_size = custom_tok.get_vocab_size()
                # Set some required attributes for HF compatibility
                self.name_or_path = custom_tok.name
                self.padding_side = "left"  # Default padding side

                # Build token-to-ID mapping for TokenMonster
                self._build_token_mappings()

                # Setup special tokens based on tokenizer type
                self._setup_special_tokens()

            def _build_token_mappings(self):
                """Build token-to-ID and ID-to-token mappings."""
                self.token_to_id_map = {}
                self.id_to_token_map = {}

                # Build mappings for all tokenizers
                for i in range(self._vocab_size):
                    try:
                        token = self.custom_tokenizer.get_token(i)
                        self.id_to_token_map[i] = token
                        self.token_to_id_map[token] = i
                    except:
                        continue

            def _setup_special_tokens(self):
                """Setup special token IDs based on tokenizer type."""
                # Default values
                self.eos_token_id = None
                self.bos_token_id = None
                self.pad_token_id = None

                if isinstance(self.custom_tokenizer, TikTokenTokenizer):
                    # GPT-4 and GPT-4o tiktoken tokenizers
                    if "gpt-4" in self.custom_tokenizer.name.lower():
                        # Try to find the actual <|endoftext|> token
                        try:
                            # For tiktoken, try to encode the special token to get its ID
                            eot_ids = self.custom_tokenizer.tokenizer.encode(
                                "<|endoftext|>"
                            )
                            if eot_ids:
                                self.eos_token_id = eot_ids[0]
                            else:
                                self.eos_token_id = 100257  # Fallback
                        except:
                            self.eos_token_id = 100257  # Standard GPT-4 endoftext token
                        self.pad_token_id = self.eos_token_id
                    else:
                        # Generic tiktoken handling
                        vocab_size = self.custom_tokenizer.get_vocab_size()
                        self.eos_token_id = vocab_size - 1
                        self.pad_token_id = self.eos_token_id

                if isinstance(self.custom_tokenizer, TokenMonsterTokenizer):
                    # Use TokenMonster's built-in UNK token
                    self.unk_token_id = self.custom_tokenizer.unk_token_id()

                    vocab_size = self.custom_tokenizer.get_vocab_size()

                    # Try to get the dictionary to search for special tokens
                    try:
                        dictionary = self.custom_tokenizer.get_dictionary()

                        # Search through the actual vocabulary for special tokens
                        for token_id, token_info in dictionary.items():
                            token_decoded = token_info.get("token_decoded", "")
                            token_raw = token_info.get("token", "")

                            # Look for EOS tokens
                            if token_decoded in [
                                "<|endoftext|>",
                                "</s>",
                                "<eos>",
                                "<end>",
                                "<|end|>",
                            ] or token_raw in [
                                "<|endoftext|>",
                                "</s>",
                                "<eos>",
                                "<end>",
                                "<|end|>",
                            ]:
                                self.eos_token_id = token_id
                                break

                            # Look for pad tokens
                            if token_decoded in [
                                "<pad>",
                                "<|pad|>",
                                "[PAD]",
                            ] or token_raw in ["<pad>", "<|pad|>", "[PAD]"]:
                                self.pad_token_id = token_id

                            # Look for BOS tokens (less common in TokenMonster)
                            if token_decoded in [
                                "<s>",
                                "<bos>",
                                "<|start|>",
                            ] or token_raw in ["<s>", "<bos>", "<|start|>"]:
                                self.bos_token_id = token_id

                    except Exception as e:
                        eval_logger.warning(
                            f"Could not access TokenMonster dictionary: {e}"
                        )

                    # Fallback strategies
                    if self.eos_token_id is None:
                        # For TokenMonster, using the last token as EOS is reasonable
                        self.eos_token_id = vocab_size - 1

                    if self.pad_token_id is None:
                        # Use EOS as pad token if no dedicated pad token found
                        self.pad_token_id = self.eos_token_id

                    # TokenMonster typically doesn't use BOS tokens
                    # Leave self.bos_token_id as None

                elif isinstance(self.custom_tokenizer, MistralTokenizer):
                    # Mistral tekken tokenizer
                    if hasattr(self.custom_tokenizer.tokenizer, "bos_id"):
                        self.bos_token_id = self.custom_tokenizer.tokenizer.bos_id
                    if hasattr(self.custom_tokenizer.tokenizer, "eos_id"):
                        self.eos_token_id = self.custom_tokenizer.tokenizer.eos_id

                    # For tekken, try to get proper special tokens
                    if self.eos_token_id is None:
                        vocab_size = self.custom_tokenizer.get_vocab_size()
                        self.eos_token_id = vocab_size - 1

                    # Use EOS as pad if no dedicated pad token
                    self.pad_token_id = self.eos_token_id if self.eos_token_id else 0

            @property
            def vocab_size(self):
                return self._vocab_size

            @property
            def pad_token(self):
                """Return pad token string if available."""
                if self.pad_token_id is not None:
                    try:
                        return self.custom_tokenizer.get_token(self.pad_token_id)
                    except:
                        return None
                return None

            @property
            def eos_token(self):
                """Return EOS token string if available."""
                if self.eos_token_id is not None:
                    try:
                        return self.custom_tokenizer.get_token(self.eos_token_id)
                    except:
                        return None
                return None

            @property
            def bos_token(self):
                """Return BOS token string if available."""
                if self.bos_token_id is not None:
                    try:
                        return self.custom_tokenizer.get_token(self.bos_token_id)
                    except:
                        return None
                return None

            @property
            def unk_token(self):
                """Return UNK token string if available."""
                if hasattr(self, "unk_token_id") and self.unk_token_id is not None:
                    try:
                        return self.custom_tokenizer.get_token(self.unk_token_id)
                    except:
                        return None
                return None

            @property
            def model_max_length(self):
                """Return a reasonable max length."""
                return 4096  # Default reasonable length

            def get_vocab(self):
                """Return vocabulary mapping."""
                return self.token_to_id_map

            def encode(self, text, add_special_tokens=True, **kwargs):
                """Encode text to token IDs."""
                print(f"\n=== TOKENIZER DEBUG ===")
                print(
                    f"Input text: '{text}' (type: {type(text)}, len: {len(text) if text else 0})"
                )
                print(f"Input repr: {repr(text)}")
                print(f"add_special_tokens: {add_special_tokens}")
                if not text:
                    eval_logger.warning(
                        f"TokenMonster encode received empty text: '{text}'"
                    )
                    return []

                try:
                    if isinstance(self.custom_tokenizer, TokenMonsterTokenizer):
                        token_ids = self.custom_tokenizer.tokenize(text)
                        if token_ids is None or len(token_ids) == 0:
                            print(
                                f"TokenMonster failed to tokenize '{text}', using fallback"
                            )

                            # Fallback strategy for numbers with leading space
                            if text.strip().isdigit() and text.startswith(" "):
                                # Try tokenizing space and number separately
                                space_tokens = self.custom_tokenizer.tokenize(" ")
                                number_tokens = self.custom_tokenizer.tokenize(
                                    text.strip()
                                )
                                print(
                                    f"Space tokens: {space_tokens}, Number tokens: {number_tokens}"
                                )
                                if space_tokens and number_tokens:
                                    return space_tokens + number_tokens

                            # Ultimate fallback: return UNK token
                            return (
                                [self.unk_token_id]
                                if hasattr(self, "unk_token_id")
                                else [0]
                            )

                    else:
                        # TikToken and Mistral return token IDs directly
                        token_ids = self.custom_tokenizer.tokenize(text)
                        if token_ids is None:
                            return []

                    # Handle add_special_tokens if needed
                    if add_special_tokens and self.bos_token_id is not None:
                        token_ids = [self.bos_token_id] + token_ids

                    return token_ids

                except Exception as e:
                    eval_logger.error(
                        f"TokenMonster encode failed for text '{text}': {e}"
                    )
                    return []

            def decode(self, token_ids, skip_special_tokens=True, **kwargs):
                """Decode token IDs to text."""
                if hasattr(token_ids, "tolist"):  # Handle torch tensors
                    token_ids = token_ids.tolist()

                # Filter special tokens if requested
                if skip_special_tokens:
                    special_tokens = {
                        self.eos_token_id,
                        self.bos_token_id,
                        self.pad_token_id,
                    }
                    special_tokens = {t for t in special_tokens if t is not None}
                    token_ids = [tid for tid in token_ids if tid not in special_tokens]

                # Convert token IDs back to text
                try:
                    if isinstance(self.custom_tokenizer, TikTokenTokenizer):
                        # TikToken can decode directly from IDs
                        try:
                            return self.custom_tokenizer.tokenizer.decode(token_ids)
                        except:
                            # Fallback to token-by-token
                            tokens = [
                                self.custom_tokenizer.get_token(tid)
                                for tid in token_ids
                                if tid < self._vocab_size
                            ]
                            return "".join(tokens)

                    elif isinstance(self.custom_tokenizer, TokenMonsterTokenizer):
                        # Use TokenMonster's built-in decode method directly
                        if not token_ids:
                            return ""
                        return self.custom_tokenizer.decode(token_ids)

                    elif isinstance(self.custom_tokenizer, MistralTokenizer):
                        # Mistral/Tekken: handle piece-based tokens
                        tokens = []
                        for tid in token_ids:
                            if tid < self._vocab_size:
                                token = self.custom_tokenizer.get_token(tid)
                                tokens.append(token)

                        # Join and handle space markers (▁ -> space)
                        text = "".join(tokens)
                        # Replace sentencepiece space marker with actual spaces
                        text = text.replace("▁", " ")
                        return text.strip()  # Remove leading/trailing spaces

                    else:
                        # Generic fallback
                        tokens = [
                            self.custom_tokenizer.get_token(tid)
                            for tid in token_ids
                            if tid < self._vocab_size
                        ]
                        return "".join(tokens)

                except Exception as e:
                    eval_logger.warning(
                        f"Failed to decode tokens {token_ids[:10]}...: {e}"
                    )
                    return ""

            def __call__(
                self,
                text,
                padding=False,
                truncation=False,
                max_length=None,
                return_tensors=None,
                add_special_tokens=True,
                **kwargs,
            ):
                """Callable interface for batch processing."""
                if isinstance(text, str):
                    texts = [text]
                    single_input = True
                else:
                    texts = text
                    single_input = False

                # Encode all texts
                all_token_ids = []
                for t in texts:
                    token_ids = self.encode(t, add_special_tokens=add_special_tokens)

                    # Handle truncation
                    if truncation and max_length and len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]

                    all_token_ids.append(token_ids)

                # Handle padding
                if padding and len(all_token_ids) > 1:
                    max_len = max(len(ids) for ids in all_token_ids)
                    if max_length:
                        max_len = min(max_len, max_length)

                    pad_id = self.pad_token_id if self.pad_token_id is not None else 0

                    for i, token_ids in enumerate(all_token_ids):
                        if len(token_ids) < max_len:
                            if self.padding_side == "left":
                                all_token_ids[i] = [pad_id] * (
                                    max_len - len(token_ids)
                                ) + token_ids
                            else:
                                all_token_ids[i] = token_ids + [pad_id] * (
                                    max_len - len(token_ids)
                                )

                # Create attention masks
                attention_masks = []
                pad_id = self.pad_token_id if self.pad_token_id is not None else 0
                for token_ids in all_token_ids:
                    mask = [1 if tid != pad_id else 0 for tid in token_ids]
                    attention_masks.append(mask)

                result = {
                    "input_ids": all_token_ids[0] if single_input else all_token_ids,
                    "attention_mask": attention_masks[0]
                    if single_input
                    else attention_masks,
                }

                # Convert to tensors if requested
                if return_tensors == "pt":
                    if single_input:
                        result["input_ids"] = torch.tensor([result["input_ids"]])
                        result["attention_mask"] = torch.tensor(
                            [result["attention_mask"]]
                        )
                    else:
                        result["input_ids"] = torch.tensor(result["input_ids"])
                        result["attention_mask"] = torch.tensor(
                            result["attention_mask"]
                        )

                return result

            def _validate_encoding(self, test_text="Hello, world!"):
                """Validate that encoding->decoding preserves the original text."""
                try:
                    token_ids = self.encode(test_text, add_special_tokens=False)
                    decoded_text = self.decode(token_ids, skip_special_tokens=True)

                    if decoded_text == test_text:
                        eval_logger.info(
                            f"Tokenizer validation passed for '{test_text}'"
                        )
                        return True
                    else:
                        eval_logger.warning(
                            f"Tokenizer validation failed:\n"
                            f"  Original: '{test_text}'\n"
                            f"  Decoded:  '{decoded_text}'\n"
                            f"  Token IDs: {token_ids}"
                        )
                        return False
                except Exception as e:
                    eval_logger.error(f"Tokenizer validation error: {e}")
                    return False

        wrapper = CustomTokenizerWrapper(custom_tokenizer)

        # Validate the tokenizer works correctly
        wrapper._validate_encoding()

        return wrapper

    def _create_tokenizer(
        self,
        pretrained: str | transformers.PreTrainedModel,
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None,
        revision: str | None = "main",
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        gguf_file: str | None = None,
        add_bos_token: bool | None = False,
        subfolder: str | None = "",
    ) -> None:
        """Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        Supports custom tokenizers: TokenMonster, TikToken, and Tekken.
        """

        # If a tokenizer object is already provided, use it directly
        if tokenizer and not isinstance(tokenizer, str):
            assert isinstance(
                tokenizer,
                (
                    transformers.PreTrainedTokenizer,
                    transformers.PreTrainedTokenizerFast,
                ),
            )
            self.tokenizer = tokenizer
            return

        # Determine tokenizer name - use explicit tokenizer if provided
        tokenizer_name = tokenizer if isinstance(tokenizer, str) else None

        # Check if we should use a custom tokenizer based on tokenizer name
        if tokenizer_name and self._should_use_custom_tokenizer(tokenizer_name):
            try:
                # Load custom tokenizer using your Tokenizer.load method
                custom_tokenizer = Tokenizer.load(tokenizer_name)

                # Create a wrapper that implements the HF tokenizer interface
                self.tokenizer = self._wrap_custom_tokenizer(custom_tokenizer)
                eval_logger.info(
                    f"Successfully loaded custom tokenizer: {tokenizer_name}"
                )
                return
            except Exception as e:
                eval_logger.error(
                    f"Failed to load custom tokenizer '{tokenizer_name}': {e}"
                )
                raise RuntimeError(
                    f"Custom tokenizer '{tokenizer_name}' could not be loaded. "
                    f"Make sure your custom tokenizer classes are properly installed and imported."
                ) from e

        # Fall back to standard HF tokenizer loading
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        if not tokenizer and gguf_file is not None:
            kwargs["gguf_file"] = gguf_file
        else:
            kwargs["use_fast"] = use_fast_tokenizer

        if add_bos_token:
            kwargs["add_bos_token"] = True

        if subfolder:
            kwargs["subfolder"] = subfolder

        # Get tokenizer based on 'pretrained' or explicit tokenizer name
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            # get the HF hub name via accessor on model
            model_name = self.model.name_or_path

        # Use tokenizer_name if provided, otherwise fall back to model_name
        target_name = tokenizer_name if tokenizer_name else model_name

        # Validate that we're not trying to load a custom tokenizer path as HF tokenizer
        if tokenizer_name and any(
            pattern in tokenizer_name for pattern in ["tokenmonster/", "tiktoken/"]
        ):
            raise ValueError(
                f"Tokenizer '{tokenizer_name}' appears to be a custom tokenizer path, "
                f"but custom tokenizers are not available. Make sure your custom tokenizer "
                f"classes are properly installed and imported."
            )

        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                target_name, **kwargs
            )
            eval_logger.info(
                f"Successfully loaded HuggingFace tokenizer: {target_name}"
            )
        except Exception as e:
            eval_logger.error(f"Failed to load tokenizer '{target_name}': {e}")
            raise RuntimeError(
                f"Could not load tokenizer '{target_name}'. "
                f"Please check that the tokenizer path is correct."
            ) from e

    def _get_accelerate_args(
        self,
        parallelize: bool | None = None,
        device_map: str | None = "auto",
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        gpus: int | None = None,
    ) -> dict:
        """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        num_machines = int(os.environ.get("WORLD_SIZE", 0)) // num_local_processes
        if (
            num_machines == 0
            and hasattr(self, "accelerator")
            and self.accelerator is not None
        ):
            eval_logger.info(
                "We are not in a distributed setting for accelerate. Setting model_parallel to False."
            )
            parallelize = False

        if parallelize is None:
            # If parallelism is unset by the user, we automatically assign model parallelism
            # if enough extra GPUs are available
            max_memory_all_gpus = get_max_memory()
            # We just want gpu, not cpu, max memory
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            parallelize = bool(num_local_processes < len(max_memory_all_gpus))
            eval_logger.info(
                f"Setting model parallel to {parallelize} since "
                f"the number of local processes is {num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )

        args = {}
        if parallelize:  # Model parallelism will be used
            max_memory = {}
            if max_memory_per_gpu is not None:  # Using the provided memory requirements
                max_memory_per_gpu_map = {
                    device_idx: max_memory_per_gpu for device_idx in range(gpus)
                }
            else:  # Estimating the possible memory requirements
                max_memory_all_gpus = get_max_memory()
                max_memory_all_gpus.pop("cpu", None)
                if hasattr(self, "accelerator"):
                    # use only 1 / num_processes of the GPUs if we are running under accelerate launch
                    max_memory_per_gpu_map = {
                        k: v
                        for k, v in max_memory_all_gpus.items()
                        if k % num_local_processes
                        == (self.accelerator.process_index % num_local_processes)
                    }
                else:
                    max_memory_per_gpu_map = max_memory_all_gpus

            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto" if device_map is None else device_map
            eval_logger.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_memory_per_gpu_map} and device map to {args.get('device_map')}"
            )

            if max_cpu_memory is not None:
                max_memory["cpu"] = max_cpu_memory

            args["offload_folder"] = offload_folder
        elif (
            device_map is None
        ):  # No model parallelism, we use the default provided device for our model
            if hasattr(self, "accelerator"):
                device_map = {"": f"{self.accelerator.device}"}
            else:
                device_map = {"": str(self.device)}
            args["max_memory"] = None
            args["device_map"] = device_map
            eval_logger.info(
                f"Model parallel was set to False, max memory was not set, and device map was set to {device_map}"
            )
        else:
            args["max_memory"] = None
            args["device_map"] = None
            eval_logger.info("Model parallel was set to False.")

        return args

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self) -> int:
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == TOKENIZER_INFINITY:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_backend(
        self,
        config: transformers.PretrainedConfig | transformers.AutoConfig,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: bool | None = False,
    ) -> None:
        """Helper method during initialization.

        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder)) model type to be used.
        sets `self.AUTO_MODEL_CLASS` appropriately if not already set.

        **If not calling HFLM.__init__() or HFLM._get_backend() within a subclass of HFLM,
        user must set `self.backend` to be either "causal" or "seq2seq" manually!**
        """

        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend in ["causal", "seq2seq"]:
                self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type", None)
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
                self.backend = "seq2seq"
                eval_logger.debug(f"Using model type '{self.backend}'")
            elif (
                getattr(config, "model_type", None) in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
                eval_logger.debug(f"Using model type '{self.backend}'")
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                        "Setting backend to causal"
                    )
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to assuming AutoModelForCausalLM
                self.backend = "causal"
                eval_logger.info(
                    f"Model type cannot be determined. Using default model type '{self.backend}'"
                )

        if self.AUTO_MODEL_CLASS is None:
            if self.backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif self.backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: str | None = None,
        subfolder: str = "",
    ) -> None:
        """Return the model config for HuggingFace models."""
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            gguf_file=gguf_file,
            subfolder=subfolder,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        # PEFT, delta weights and quantization options
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        quantization_config: AutoQuantizationConfig | None = None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        """Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs or {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map"),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        if not autogptq and not gptqmodel:
            if model_kwargs.get("load_in_4bit"):
                assert vparse(transformers.__version__) >= vparse("4.30.0"), (
                    "load_in_4bit requires transformers >= 4.30.0"
                )
                if compute_dtype := model_kwargs.get("bnb_4bit_compute_dtype"):
                    model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(compute_dtype)

            self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
                quantization_config=quantization_config,
                subfolder=subfolder,
                **model_kwargs,
            )
        else:
            if autogptq and gptqmodel:
                raise ValueError(
                    "Cannot use both 'autogptq' and 'gptqmodel' options at the same time."
                )

            if autogptq:
                try:
                    from auto_gptq import AutoGPTQForCausalLM
                except ModuleNotFoundError as exception:
                    raise type(exception)(
                        "Tried to load auto_gptq, but auto-gptq is not installed ",
                        "please install auto-gptq via pip install lm-eval[gptq] or pip install -e .[gptq]",
                    ) from exception

                self._model = AutoGPTQForCausalLM.from_quantized(
                    pretrained,
                    trust_remote_code=trust_remote_code,
                    model_basename=None if autogptq is True else Path(autogptq).stem,
                    use_safetensors=True
                    if autogptq is True
                    else autogptq.endswith(".safetensors"),
                    **model_kwargs,
                )

            if gptqmodel:
                try:
                    from gptqmodel import GPTQModel
                except ModuleNotFoundError as exception:
                    raise type(exception)(
                        "Tried to load gptqmodel, but gptqmodel is not installed ",
                        "please install gptqmodel via `pip install gptqmodel --no-build-isolation` or `pip install lm-eval[gptqmodel] --no-build-isolation`",
                    ) from exception

                self._model = GPTQModel.from_quantized(
                    pretrained, trust_remote_code=trust_remote_code, **model_kwargs
                )

        if peft and delta:
            raise ValueError(
                "Cannot use both 'peft' and 'delta' options at the same time."
            )

        if peft:
            from peft import PeftModel
            from peft import __version__ as PEFT_VERSION

            if model_kwargs.get("load_in_4bit") and vparse(PEFT_VERSION) < vparse(
                "0.4.0"
            ):
                raise AssertionError("load_in_4bit requires peft >= 0.4.0")

            # Compatible with Gemma3 (multimodal) and old models
            if hasattr(self._model.config, "text_config") and hasattr(
                self._model.config.text_config, "vocab_size"
            ):
                vocab_size = self._model.config.text_config.vocab_size
            else:
                vocab_size = self._model.config.vocab_size

            if vocab_size != len(self.tokenizer):
                # resize model for LoRAs with added tokens
                eval_logger.info(
                    f"Model config indicates vocab_size='{vocab_size}', but found tokenizer with vocab size '{len(self.tokenizer)}'. Resizing model embedding layer..."
                )
                self._model.resize_token_embeddings(len(self.tokenizer))
            self._model = PeftModel.from_pretrained(
                self._model, peft, revision=revision
            )
        elif delta:
            if autogptq:
                eval_logger.warning(
                    "Delta weights might trigger unexpected behavior when used with AutoGPTQ."
                )
            _model_delta = self.AUTO_MODEL_CLASS.from_pretrained(
                delta,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            for name, param in self._model.state_dict().items():
                try:
                    param.data += _model_delta.state_dict()[name]
                except KeyError as e:
                    raise KeyError(
                        f"Delta model is missing weights for layer: {name}"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to add delta weights to layer {name}. Error: {e}"
                    ) from e

            del _model_delta

    def _detect_batch_size(self, requests: Sequence | None = None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size: int):
            if self.backend == "seq2seq":
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
            for _ in range(5):
                out = F.log_softmax(  # noqa: F841
                    self._model_call(test_batch, **call_kwargs),
                    dim=-1,
                    dtype=self.softmax_dtype,
                )

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            clear_torch_cache()
            return batch_size

        clear_torch_cache()
        return batch_size

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warning(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens: Iterator[list[str]], skip_special_tokens: bool = True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """

        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits

            assert self.AUTO_MODEL_CLASS in (
                transformers.AutoModelForCausalLM,
                transformers.AutoModelForVision2Seq,
            )
            return self.model(inps).logits

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )

    def _select_cont_toks(
        self,
        logits: torch.Tensor,
        contlen: int | None = None,
        inplen: int | None = None,
    ) -> torch.Tensor:
        if self.backend == "causal":
            assert contlen and inplen, (
                "Must pass input len and cont. len to select scored logits for causal LM"
            )
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif self.backend == "seq2seq":
            assert contlen and not inplen, (
                "Selecting scored logits for Seq2SeqLM requires only cont. len"
            )
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: list[tuple[list[int], list[int]]] = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        # Handle distributed case padding
        pad_amnt = 0
        if self.world_size > 1:
            mytensor = torch.tensor(len(all_windows), device=self.device)
            gathered = self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
            pad_amnt = max(gathered) - gathered[self.rank]
            if pad_amnt > 0:
                all_windows += pad_amnt * [all_windows[0]]

        all_nlls = []
        batch_size = adaptive_batch_size or self.batch_size
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
                override_bs=len(batch_windows),
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Remove padding if necessary
        if (self.world_size > 1) and (pad_amnt > 0):
            all_nlls = all_nlls[:-pad_amnt]

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key for the sorted method."""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key to group and lookup one-token continuations."""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        print(f"\n=== PROCESSING {len(requests)} LOGLIKELIHOOD REQUESTS ===")

        # Debug: Show first few requests to understand the data
        empty_continuation_count = 0
        for i, req in enumerate(requests[:5]):  # Show first 5 requests
            context_str, continuation_str = req[0]
            context_enc, continuation_enc = req[1], req[2]
            print(f"Request {i}:")
            print(
                f"  Context: '{context_str}' -> tokens: {context_enc[:10]}{'...' if len(context_enc) > 10 else ''} (len: {len(context_enc)})"
            )
            print(
                f"  Continuation: '{continuation_str}' -> tokens: {continuation_enc} (len: {len(continuation_enc)})"
            )
            if len(continuation_enc) == 0:
                print(f"  *** EMPTY CONTINUATION DETECTED! ***")
                empty_continuation_count += 1

        # Count total empty continuations
        total_empty = sum(1 for req in requests if len(req[2]) == 0)
        print(f"Total empty continuations in batch: {total_empty}/{len(requests)}")
        print(f"=== END REQUEST PREVIEW ===\n")
        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.backend == "causal" and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for request_data, context_enc, continuation_enc in chunk:
                context_str, continuation_str = request_data

                # print(f"\n--- Processing chunk item ---")
                # print(f"Context: '{context_str}' (tokens: {len(context_enc)})")
                # print(f"Continuation: '{continuation_str}' (tokens: {len(continuation_enc)})")

                # Safety check for empty continuations
                if len(continuation_enc) == 0:
                    eval_logger.warning(
                        f"Empty continuation detected. Context: '{context_str}', Continuation: '{continuation_str}'"
                    )

                    # Create a dummy continuation with UNK token to maintain pipeline consistency
                    continuation_enc = [
                        self.tokenizer.unk_token_id
                        if hasattr(self.tokenizer, "unk_token_id")
                        and self.tokenizer.unk_token_id is not None
                        else 0
                    ]
                    eval_logger.warning(
                        f"Using UNK token {continuation_enc[0]} as fallback for empty continuation"
                    )

                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.backend == "causal":
                    total_length = len(context_enc) + len(continuation_enc)
                    if total_length > self.max_length + 1:
                        eval_logger.warning(
                            f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                            f"exceeds model's maximum length ({self.max_length}). "
                            f"Truncating {total_length - self.max_length + 1} tokens from the left."
                        )
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.backend == "seq2seq":
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.backend == "causal":
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.backend == "seq2seq":
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs),
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.backend == "causal"
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(  # noqa
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    # Use trailing slice [-cont_toks.shape[1]:] to handle variable length cont_len (but same ctx+cont[:-1]).
                    # i.e. continuations can be sliced at diff points. Collator ensures we have sufficient greedy_tokens
                    # by choosing key with longest cont if group_by="contexts".
                    max_equal = (
                        greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks
                    ).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        res = []

        def _collate(req: tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise TypeError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs:
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert max_ctx_len > 0, (
                    f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
                )
            elif self.backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.backend == "causal":
                    cont_toks = cont_toks[context_enc.shape[1] :]

                # Handle integer think_end_token: find last occurrence and strip tokens after it
                if isinstance(self.think_end_token, int):
                    think_token_indices = [
                        i
                        for i, token in enumerate(cont_toks)
                        if token == self.think_end_token
                    ]
                    if think_token_indices:
                        cont_toks = cont_toks[think_token_indices[-1] + 1 :]

                s = self.tok_decode(cont_toks)

                # Strip leading whitespace if we removed thinking tokens
                if isinstance(self.think_end_token, int):
                    s = s.lstrip()

                # Apply post-processing: remove stop sequences and string-based thinking tokens
                s = postprocess_generated_text(
                    generation=s,
                    stop=until,
                    think_end_token=self.think_end_token
                    if isinstance(self.think_end_token, str)
                    else None,
                )
                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Method to apply a chat template to a list of chat history between user and model."""
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                **self.chat_template_args,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                **self.chat_template_args,
            )

        return chat_templated

    def get_model_info(self) -> dict:
        """Method to get Hugging Face model information for experiment reproducibility."""

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        def get_model_sha(pretrained: str, revision: str) -> str:
            try:
                model_info = HfApi().model_info(repo_id=pretrained, revision=revision)
                return model_info.sha
            except Exception as e:
                eval_logger.debug(
                    f"Failed to get model SHA for {pretrained} at revision {revision}. Error: {e}"
                )
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
            "model_sha": get_model_sha(self.pretrained, self.revision),
        }
        if self.peft:
            model_info["peft_sha"] = get_model_sha(self.peft, self.revision)
        if self.delta:
            model_info["delta_sha"] = get_model_sha(self.delta, self.revision)
        return model_info
