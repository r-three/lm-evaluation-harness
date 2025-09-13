import argparse
import collections
import enum
import functools
import json
import logging
import operator as op
import os
import random
import re
import sys
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import tokenizers
import transformers
from transformers import AutoTokenizer
from typing_extensions import TypeAlias


Vocab = dict[str, list[int]]

LOG = logging.getLogger(__name__)

random.seed(42)  # Set the seed to a fixed value

ALIGNED_BOS = "~SPECIAL~ALIGNED~BOS~SYMBOL~"

TOKENIZER_NAMES = {
    "BERT multilingual base model (cased)": "google-bert/bert-base-multilingual-cased",
    "BERT base model (uncased)": "google-bert/bert-base-uncased",
    "T5": "google-t5/t5-base",
    "mT5": "google/mt5-base",
    "XGLM-564M": "facebook/xglm-564M",
    "Gemma 2": "google/gemma-2-2b",
    "Phi-3-Mini-4K-Instruct": "microsoft/Phi-3-mini-4k-instruct",
    "Mistral v3": "mistralai/Mistral-7B-Instruct-v0.3",
    "TokenMonster": "tokenmonster/english-32000-balanced-v1",
    "ByT5 - Small": "google/byt5-small",
    "BLOOM": "bigscience/bloom",
    "GPT-2": "gpt2",
    "GPT-4 Tiktoken": "tiktoken/gpt-4",
    "GPT-4o Tiktoken": "tiktoken/gpt-4o",
    "Mistral v3 (tekken)": "mistralai/tekken",  # Use this link to call it: https://docs.mistral.ai/guides/tokenization/
    "Llama-3.2 1B ": "meta-llama/Llama-3.2-1B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Aya Expanse 8B": "CohereLabs/aya-expanse-8b",
    "Common Pile v1.0": "common-pile/comma-v0.1",
}

TOKENIZER_TYPES = {
    "BERT multilingual base model (cased)": "WordPiece",
    "BERT base model (uncased)": "WordPiece",
    "T5": "SentencePiece_Unigram",
    "mT5": "SentencePiece_Unigram",
    "XGLM-564M": "SentencePiece_Unigram",
    "Gemma 2": "BPE",  # BPE in HF but Originally SentencePiece_Unigram
    "Phi-3-Mini-4K-Instruct": "SentencePiece_BPE",
    "Mistral v3": "SentencePiece_BPE",
    "TokenMonster": "",
    "ByT5 - Small": "byte-level",
    "BLOOM": "BPE",
    "GPT-2": "BPE",
    "GPT-4 Tiktoken": "BPE",
    "GPT-4o Tiktoken": "BPE",
    "Mistral v3 (tekken)": "BPE",  # Use this link to call it: https://docs.mistral.ai/guides/tokenization/
    "Llama-3.2 1B ": "BPE",
    "Qwen3-8B": "BPE",
    "Aya Expanse 8B": "SentencePiece",
    "Common Pile v1.0": "BPE",
}

TOKENIZER_N_SPECIAL_TOKENS_PER_WORD = {
    "BERT multilingual base model (cased)": 2,  # Example: ['[CLS]', 'Families', '[SEP]']
    "BERT base model (uncased)": 2,  # Example: ['[CLS]', 'families', '[SEP]']
    "T5": 1,  # Example: ['▁Familie', 's', '</s>']
    "mT5": 1,  # Example: ['▁Familie', 's', '</s>']
    "XGLM-564M": 1,  # Example: ['▁Familie', 's', '</s>']
    "Gemma 2": 1,  # Example: ['<bos>', 'Families']
    "Phi-3-Mini-4K-Instruct": 0,  # Example: ['▁Famil', 'ies']
    "Mistral v3": 1,  # Example: ['<s>', '▁Famil', 'ies']
    "TokenMonster": 0,  # Example: [np.uint16(586), np.uint16(17496)]
    "ByT5 - Small": 1,  # Example: [73, 100, 112, 108, 111, 108, 104, 118, 1]
    "BLOOM": 0,  # Example: ['Famil', 'ies']
    "GPT-2": 0,  # Example: ['F', 'am', 'ilies']
    "GPT-4 Tiktoken": 0,  # Example: [37, 60004]
    "GPT-4o Tiktoken": 0,  # Example: [139342]
    "Mistral v3 (tekken)": 0,  # Example: [109925, 1564]
    "Llama-3.2 1B ": 1,  # Example: ['<|begin_of_text|>', 'F', 'amilies']
    "Qwen3-8B": 0,  # Example: ['F', 'amilies']
    "Aya Expanse 8B": 1,  # Example: ['<BOS_TOKEN>', 'Families']
    "Common Pile v1.0": 0,  # Example: ['F', 'amil', 'ies']
}

LANGUAGE_KEYS = {
    "sentence_eng_Latn": "eng_Latn",  # english
    "sentence_zho_Hans": "zho_Hani",  # chinese
    "sentence_tur_Latn": "tur_Latn",  # turkish
    "sentence_pes_Arab": "fas_Arab",  # persian
    "sentence_ita_Latn": "ita_Latn",  # Italian
}


def hf_load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a tokenizer for the specified model.
    If tokenizer available locally, uses the local path"""
    kwargs = dict()
    if "aya" in model_name.lower():
        kwargs["use_fast"] = True

    model_path = model_name
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


BYTES_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTES = {v: k for k, v in BYTES_TO_UNICODE.items()}


def real_unicode(word: str) -> str:
    bytes_word = []
    for c in word:
        if c != " ":
            if c in UNICODE_TO_BYTES:
                c = chr(UNICODE_TO_BYTES[c])
        bytes_word.append(c.encode("utf-8"))
    return b"".join(bytes_word).decode("utf-8")


def to_bytes(s: bytes | str | int) -> bytes:
    if isinstance(s, str):
        s = s.encode("utf-8")
    if isinstance(s, int):
        s = bytes([s])
    # Now s is def bytes
    return s


def join_vocabs(vocabs: dict[str, Vocab]) -> Vocab:
    joint = functools.reduce(op.or_, [v.keys() for v in vocabs.values()])
    return {s: i for i, s in enumerate(sorted(joint, key=to_bytes))}


class Tokenizer:
    """Tokenizer wrapper that unifies interface."""

    def __init__(self, name: str, tokenizer):
        self._name = name
        self.tokenizer = tokenizer

    @property
    def name(self):
        return self._name

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_vocab(self):
        raise NotImplementedError

    def get_token(self, i):
        raise NotImplementedError

    def get_bos_str(self):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError

    def tokenize(self, input_text):
        raise NotImplementedError

    @classmethod
    def load(cls, name):
        if name.startswith("tokenmonster"):
            return TokenMonsterTokenizer.load(name)
        if name.startswith("tiktoken"):
            return TikTokenTokenizer.load(name)
        if "tekken" in name:
            return MistralTokenizer.load(name)
        return HFTokenizer.load(name)


class HFTokenizer(Tokenizer):
    def __init__(self, *args, bos_str: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bos_str = bos_str

    def info(self):
        return {"data": {"tokenizer": {"name": "huggingface", "path": self.name}}}

    def get_vocab_size(self):
        if "byt5" in self.name:
            return self.tokenizer.vocab_size
        return self.tokenizer.get_vocab_size()

    def get_token(self, i):
        if "byt5" in self.name:
            token = self.tokenizer.convert_ids_to_tokens(i)
            # We are a special value.
            if len(token) > 1:
                return token
            as_int = ord(token)
            as_bytes = bytes([as_int])
            try:
                return as_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return as_int  # as_bytes
        t = self.tokenizer.id_to_token(i)
        if t == self.bos_str:
            return ALIGNED_BOS
        if isinstance(self.tokenizer.model, tokenizers.models.WordPiece):
            # If it is not a continuation character, then it is the start of a word. Other tokenizers start the word with a subword token that has a space to start.
            if not t.startswith("##"):
                return f" {t}"
            return re.sub(r"##([^#])", r"\1", t)
        if isinstance(self.tokenizer.model, tokenizers.models.Unigram) or any(
            n in self.name for n in ("gemma", "Phi-3", "Mistral-7B-Instruct-v0.3")
        ):
            # Replace whitespace handling with actual whitespace.
            return t.replace("▁", " ")
        # BPE models.
        return real_unicode(t)

    def get_vocab(self):  # TODO
        # Track multiple values because tekken and tokenmonster are weird
        vocab = collections.defaultdict(list)
        for i in range(self.get_vocab_size()):
            vocab[to_bytes(self.get_token(i))].append(i)
        if len(vocab) != self.get_vocab_size():
            logging.error(
                "Built vocab size (%d) does not match declared vocab size (%d) for %s",
                len(vocab),
                self.get_vocab_size(),
                self.info()["data"]["tokenizer"]["name"],
            )
        return vocab

    def tokenize(self, input_text):  # TODO
        encoded_output = self.tokenizer.encode(input_text)
        if hasattr(encoded_output, "tokens"):  # Case: tokenizers.Tokenizer object
            return encoded_output.tokens
        elif isinstance(encoded_output, list):  # Case: already a list of strings
            return encoded_output
        else:
            raise ValueError("Unexpected return type from tokenizer.encode()")

    @classmethod
    def load(cls, name):
        try:
            tok = hf_load_tokenizer(name)
        except:
            tok = transformers.AutoTokenizer.from_pretrained(name)
        sts = getattr(tok, "special_tokens_map", {})
        if "bert" in name:
            bos_str = sts.get("cls_token")
        elif "t5" in name:
            bos_str = sts.get("pad_token")
        else:
            bos_str = sts.get("bos_token")
        if hasattr(tok, "_tokenizer"):
            tok = tok._tokenizer
        return cls(name, tok, bos_str=bos_str)


# Note, GPT4 and GPT4o don't have BOS
class TikTokenTokenizer(Tokenizer):
    def info(self):
        return {
            "data": {"tokenizer": {"name": "tiktoken", "path": self.name.split("/")[1]}}
        }

    def get_token(self, i):
        try:
            b = self.tokenizer.decode_single_token_bytes(i)
        except KeyError:
            return f"~~~~~undefined {i}~~~~~~"
        return b.decode("latin-1")

    def get_vocab_size(self):
        return self.tokenizer.n_vocab

    def get_vocab(self):  # TODO
        # Track multiple values because tekken and tokenmonster are weird
        vocab = collections.defaultdict(list)
        for i in range(self.get_vocab_size()):
            vocab[to_bytes(self.get_token(i))].append(i)
        if len(vocab) != self.get_vocab_size():
            logging.error(
                "Built vocab size (%d) does not match declared vocab size (%d) for %s",
                len(vocab),
                self.get_vocab_size(),
                self.info()["data"]["tokenizer"]["name"],
            )
        return vocab

    def tokenize(self, input_text):  # TODO
        return self.tokenizer.encode(input_text)

    @classmethod
    def load(cls, name):
        import tiktoken

        tok = tiktoken.encoding_for_model(name.split("/")[1])
        return cls(name, tok)


class TokenMonsterTokenizer(Tokenizer):
    def info(self):
        return {
            "data": {
                "tokenizer": {"name": "tokenmonster", "path": self.name.split("/")[1]}
            }
        }

    def get_token(self, i):
        return self.tokenizer.id_to_token(i)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_vocab(self):  # TODO
        """Version that excludes ALL duplicate tokens, not just replacement chars"""
        vocab = collections.defaultdict(list)

        # Build full vocabulary first
        for i in range(self.get_vocab_size()):
            vocab[to_bytes(self.get_token(i))].append(i)

        # Find duplicates
        duplicates = {
            byte_seq: token_ids
            for byte_seq, token_ids in vocab.items()
            if len(token_ids) > 1
        }

        if duplicates:
            print(f"Found {len(duplicates)} duplicate byte sequences")
            for byte_seq, token_ids in duplicates.items():
                char_repr = byte_seq.decode("utf-8", errors="replace")
                print(f"  '{char_repr}': {len(token_ids)} duplicates")

        # Build clean vocab keeping only first occurrence of each duplicate
        clean_vocab = {}
        total_excluded = 0

        for byte_seq, token_ids in vocab.items():
            if len(token_ids) > 1:
                # Keep only the first token ID for duplicates
                clean_vocab[byte_seq] = [token_ids[0]]
                total_excluded += len(token_ids) - 1
            else:
                # Keep single tokens as-is
                clean_vocab[byte_seq] = token_ids

        print(
            f"Strict filtering: {len(clean_vocab)} unique tokens ({total_excluded} duplicates excluded) for {self.info()}"
        )
        return clean_vocab

    def tokenize(self, input_text):  # TODO
        return [int(t) for t in self.tokenizer.tokenize(input_text)]

    @classmethod
    def load(cls, name):
        import tokenmonster

        tok = tokenmonster.load(name.split("/")[1])
        return cls(name, tok)


class MistralTokenizer(Tokenizer):
    def info(self):
        return {"data": {"tokenizer": {"name": "tekken", "path": "tekken"}}}

    def get_token(self, i):
        if i == self.tokenizer.bos_id:
            return ALIGNED_BOS
        return self.tokenizer.id_to_piece(i)

    def get_vocab_size(self):
        return self.tokenizer.n_words

    def get_vocab(self):  # TODO
        """Version that excludes ALL duplicate tokens, not just replacement chars"""
        vocab = collections.defaultdict(list)

        # Build full vocabulary first
        for i in range(self.get_vocab_size()):
            vocab[to_bytes(self.get_token(i))].append(i)

        # Find duplicates
        duplicates = {
            byte_seq: token_ids
            for byte_seq, token_ids in vocab.items()
            if len(token_ids) > 1
        }

        if duplicates:
            print(f"Found {len(duplicates)} duplicate byte sequences")
            for byte_seq, token_ids in duplicates.items():
                char_repr = byte_seq.decode("utf-8", errors="replace")
                print(f"  '{char_repr}': {len(token_ids)} duplicates")

        # Build clean vocab keeping only first occurrence of each duplicate
        clean_vocab = {}
        total_excluded = 0

        for byte_seq, token_ids in vocab.items():
            if len(token_ids) > 1:
                # Keep only the first token ID for duplicates
                clean_vocab[byte_seq] = [token_ids[0]]
                total_excluded += len(token_ids) - 1
            else:
                # Keep single tokens as-is
                clean_vocab[byte_seq] = token_ids

        print(
            f"Strict filtering: {len(clean_vocab)} unique tokens ({total_excluded} duplicates excluded) for {self.info()}"
        )
        return clean_vocab

    def tokenize(self, input_text):  # TODO
        return self.tokenizer.encode(input_text, False, False)

    @classmethod
    def load(cls, name):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tok = MistralTokenizer.v3(is_tekken=True)
        tok = tok.instruct_tokenizer.tokenizer
        return cls(name, tok)
