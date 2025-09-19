## Directly taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/e1a7a39c7f08eeff38880cf9a3a07e1390f86d63/lm_eval/tasks/wikitext/preprocess_wikitext.py

import re


def wikitext_detokenizer(doc):
    string = doc["text"]
    # Handle byte-encoded text
    if isinstance(string, str) and "\\x" in string:
        try:
            # Convert escaped hex bytes back to proper UTF-8
            # First convert string representation back to bytes
            string_bytes = (
                string.encode("utf-8").decode("unicode_escape").encode("latin1")
            )
            string = string_bytes.decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass  # Keep original if decoding fails

    # Remove the special tokens
    string = string.replace("*START*ARTICLE_", "")
    string = string.replace("*START*SECTION_", "")
    string = string.replace("*START*PARAGRAPH_", "")
    string = string.replace("_NEWLINE_", "\n")
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def process_results(doc, results):
    (loglikelihood,) = results

    # Get the raw text
    text = doc["text"]

    # Clean the text the same way as in your detokenizer
    # Remove special wiki-40b tokens
    clean_text = text.replace("*START*ARTICLE_", "")
    clean_text = clean_text.replace("*START*SECTION_", "")
    clean_text = clean_text.replace("*START*PARAGRAPH_", "")
    clean_text = clean_text.replace("_NEWLINE_", "\n")

    # Handle byte-encoded text if present
    if "\\x" in clean_text:
        try:
            import codecs

            clean_text = codecs.decode(clean_text, "unicode_escape")
        except:
            pass  # Keep original if decoding fails

    clean_text = clean_text.strip()

    # Count words and bytes on the CLEANED text
    _words = len(re.split(r"\s+", clean_text)) if clean_text else 0
    _bytes = len(clean_text.encode("utf-8"))

    return {
        # "perplexity": (loglikelihood, _tokens),
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
