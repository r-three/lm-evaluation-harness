import re
import jieba
import string
from typing import Dict, Any, Tuple, List


class ChineseTextPreprocessor:
    """
    Enhanced Chinese text preprocessor using jieba segmenter
    """
    
    def __init__(self, use_paddle: bool = False, enable_parallel: bool = True):
        """
        Initialize the preprocessor
        
        Args:
            use_paddle: Whether to use Paddle mode for better accuracy (requires paddlepaddle)
            enable_parallel: Whether to enable parallel processing for jieba
        """
        self.use_paddle = use_paddle
        
        # Enable paddle mode if requested and available
        if use_paddle:
            try:
                import paddle
                jieba.enable_paddle()
            except ImportError:
                print("Warning: paddlepaddle not installed, falling back to default mode")
                self.use_paddle = False
        
        # Enable parallel processing
        if enable_parallel:
            jieba.enable_parallel()
        
        # Chinese punctuation mapping
        self.chinese_punctuation_map = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '；': ';',
            '：': ':',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '《': '<',
            '》': '>',
            '—': '-',
            '…': '...',
            '～': '~'
        }
        
        # Regex patterns for common issues
        self.patterns = {
            'excessive_whitespace': re.compile(r'\s+'),
            'mixed_punctuation': re.compile(r'([，。！？；：])\s+'),
            'english_chinese_space': re.compile(r'([a-zA-Z0-9])\s*([一-龯])|([一-龯])\s*([a-zA-Z0-9])'),
            'brackets_space': re.compile(r'([（【《""''])\s*([^）】》""'']*?)\s*([）】》""''])'),
            'number_units': re.compile(r'(\d+)\s*(年|月|日|时|分|秒|元|米|公里|千米|克|千克|公斤)'),
            'repeated_punctuation': re.compile(r'([。！？，；：]){2,}')
        }

    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize Chinese punctuation to standard forms
        """
        for chinese_punct, english_punct in self.chinese_punctuation_map.items():
            # Keep Chinese punctuation but ensure consistent spacing
            text = text.replace(chinese_punct, chinese_punct)
        
        # Handle mixed punctuation spacing
        text = self.patterns['mixed_punctuation'].sub(r'\1', text)
        
        return text

    def handle_chinese_english_mixing(self, text: str) -> str:
        """
        Properly handle spacing between Chinese and English text
        """
        # Add space between English and Chinese characters
        def replace_mixed(match):
            if match.group(1) and match.group(2):  # English before Chinese
                return f"{match.group(1)} {match.group(2)}"
            elif match.group(3) and match.group(4):  # Chinese before English
                return f"{match.group(3)} {match.group(4)}"
            return match.group(0)
        
        return self.patterns['english_chinese_space'].sub(replace_mixed, text)

    def clean_brackets_and_quotes(self, text: str) -> str:
        """
        Clean spacing within brackets and quotes
        """
        return self.patterns['brackets_space'].sub(r'\1\2\3', text)

    def normalize_numbers_and_units(self, text: str) -> str:
        """
        Normalize spacing between numbers and Chinese units
        """
        return self.patterns['number_units'].sub(r'\1\2', text)

    def remove_redundant_punctuation(self, text: str) -> str:
        """
        Remove repeated punctuation marks
        """
        return self.patterns['repeated_punctuation'].sub(r'\1', text)

    def chinese_wikitext_detokenizer(self, doc: Dict[str, Any]) -> str:
        """
        Enhanced detokenizer specifically for Chinese WikiText
        """
        text = doc["text"]
        
        # Apply original English-focused cleaning first (for mixed content)
        text = self.apply_original_cleaning(text)
        
        # Apply Chinese-specific preprocessing
        text = self.normalize_punctuation(text)
        text = self.handle_chinese_english_mixing(text)
        text = self.clean_brackets_and_quotes(text)
        text = self.normalize_numbers_and_units(text)
        text = self.remove_redundant_punctuation(text)
        
        # Final whitespace cleanup
        text = self.patterns['excessive_whitespace'].sub(' ', text)
        text = text.strip()
        
        return text

    def apply_original_cleaning(self, text: str) -> str:
        """
        Apply the original wikitext cleaning for mixed English content
        """
        # contractions
        text = text.replace("s '", "s'")
        text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)
        # number separators
        text = text.replace(" @-@ ", "-")
        text = text.replace(" @,@ ", ",")
        text = text.replace(" @.@ ", ".")
        # punctuation
        text = text.replace(" : ", ": ")
        text = text.replace(" ; ", "; ")
        text = text.replace(" . ", ". ")
        text = text.replace(" ! ", "! ")
        text = text.replace(" ? ", "? ")
        text = text.replace(" , ", ", ")
        # double brackets
        text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
        text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
        text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
        text = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', text)
        text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
        # miscellaneous
        text = text.replace("= = = =", "====")
        text = text.replace("= = =", "===")
        text = text.replace("= =", "==")
        text = text.replace(" " + chr(176) + " ", chr(176))
        text = text.replace(" \n", "\n")
        text = text.replace("\n ", "\n")
        text = text.replace(" 's", "'s")
        
        return text

    def segment_chinese_text(self, text: str, cut_all: bool = False) -> List[str]:
        """
        Segment Chinese text using jieba
        
        Args:
            text: Input text
            cut_all: Whether to use full mode (cut_all=True) or precise mode
            
        Returns:
            List of segmented words
        """
        if self.use_paddle:
            return jieba.cut(text, use_paddle=True)
        else:
            return jieba.cut(text, cut_all=cut_all)

    def count_chinese_words(self, text: str) -> int:
        """
        Count words in Chinese text using jieba segmentation
        """
        # Clean the text first
        cleaned_text = self.patterns['excessive_whitespace'].sub(' ', text).strip()
        
        # Segment the text
        words = list(self.segment_chinese_text(cleaned_text))
        
        # Filter out pure whitespace and punctuation
        meaningful_words = [
            word for word in words 
            if word.strip() and not all(char in string.punctuation + '，。！？；：""''（）【】《》—…～' for char in word)
        ]
        
        return len(meaningful_words)

    def process_results(self, doc: Dict[str, Any], results: Tuple) -> Dict[str, Tuple]:
        """
        Enhanced process_results function for Chinese text
        """
        (loglikelihood,) = results
        
        # Get original text
        original_text = doc["text"]
        
        # Count words using jieba segmentation instead of simple whitespace split
        word_count = self.count_chinese_words(original_text)
        
        # Count bytes (same as original)
        byte_count = len(original_text.encode("utf-8"))
        
        
        return {
            "word_perplexity": (loglikelihood, word_count),
            "byte_perplexity": (loglikelihood, byte_count),
            "bits_per_byte": (loglikelihood, byte_count)
        }


# Convenience functions for backward compatibility
def chinese_wikitext_detokenizer(doc: Dict[str, Any]) -> str:
    """
    Convenience function that uses default preprocessor settings
    """
    preprocessor = ChineseTextPreprocessor()
    return preprocessor.chinese_wikitext_detokenizer(doc)


def process_results(doc: Dict[str, Any], results: Tuple) -> Dict[str, Tuple]:
    """
    Convenience function that uses default preprocessor settings
    """
    preprocessor = ChineseTextPreprocessor()
    return preprocessor.process_results(doc, results)
