"""Text preprocessing module for tokenization, normalization, and filtering."""
import re
from typing import List, Optional, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Basic alphanumeric token pattern matching words, numbers, and contractions
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


class TextPreprocessor:
    """Preprocesses queries and documents for lexical retrieval.

    By default, performs lowercasing and regex-based tokenization without
    aggressive stemming or stopword deletion. This preserves essential question
    particles ('who', 'what', 'when') that are vital for QA passage retrieval,
    relying on BM25's IDF component to naturally down-weight non-informative terms.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        stemming: bool = False,
        custom_stopwords: Optional[Set[str]] = None,
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming

        self.stopwords: Set[str] = custom_stopwords or set()
        if self.remove_stopwords and not self.stopwords:
            try:
                import nltk
                from nltk.corpus import stopwords
                try:
                    self.stopwords = set(stopwords.words("english"))
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                    self.stopwords = set(stopwords.words("english"))
            except ImportError:
                logger.warning("NLTK not installed; stopword removal disabled.")
                self.remove_stopwords = False

        self.stemmer = None
        if self.stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                logger.warning("NLTK not installed; stemming disabled.")
                self.stemming = False

    def tokenize(self, text: str) -> List[str]:
        """Converts raw input text into a list of normalized tokens.

        Args:
            text: Raw input string.

        Returns:
            List of string tokens.
        """
        if not text or not isinstance(text, str):
            return []

        if self.lowercase:
            text = text.lower()

        tokens = TOKEN_PATTERN.findall(text)

        if self.remove_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        if self.stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens
