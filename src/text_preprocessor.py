# File: src/text_preprocessor.py
import re
import string

class AmharicPreprocessor:
    """
    A class to preprocess Amharic text by normalizing characters,
    cleaning irrelevant symbols, and tokenizing the text.
    """

    def __init__(self):
        # Dictionary for character normalization
        self.normalization_dict = {
            'ሀ': 'ሃ', 'ሐ': 'ሃ', 'ኀ': 'ሃ', 'ኻ': 'ሃ',
            'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሦ': 'ሶ', 'ሧ': 'ሷ',
            'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ፂ': 'ጺ', 'ፃ': 'ጻ', 'ፄ': 'ጼ', 'ፅ': 'ጽ', 'ፆ': 'ጾ',
            'አ': 'ኣ', 'ዐ': 'ኣ',
            'ዉ': 'ው', 'ዴ': 'ደ', 'ቺ': 'ች'
            # Add other common variations as you find them
        }

    def normalize_text(self, text: str) -> str:
        """
        Normalizes Amharic characters to a consistent form.
        Example: 'ሰላም' and 'ሠላም' both become 'ሰላም'.
        """
        for char, replacement in self.normalization_dict.items():
            text = text.replace(char, replacement)
        return text

    def clean_text(self, text: str, remove_punc=True) -> str:
        """
        Cleans the text by removing URLs, mentions, hashtags, and optionally, punctuation.
        """
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (#hashtag)
        text = re.sub(r'#\w+', '', text)
        
        if remove_punc:
            # Amharic and common English punctuation to remove for tokenization
            # Note: We keep some punctuation if they are part of a price or product
            # but for general tokenization, removing them is standard.
            amharic_punc = "።፣፤፥፦፧፨"
            english_punc = string.punctuation.replace('.', '').replace('-', '') # Keep dots and hyphens for now
            all_punc = amharic_punc + english_punc
            translator = str.maketrans('', '', all_punc)
            text = text.translate(translator)
            
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """
        Splits the text into a list of tokens (words).
        A simple split by space is a strong baseline for Amharic.
        """
        if not text:
            return []
        return text.split(' ')

    def preprocess(self, text: str) -> list[str]:
        """
        A full pipeline that applies normalization, cleaning, and tokenization.
        This is the main method you will use.
        """
        normalized_text = self.normalize_text(text)
        # For NER, we often don't want to remove all punctuation initially
        # as it might be part of an entity (e.g., price).
        # We will do a lighter cleaning first.
        cleaned_text = self.clean_text(normalized_text, remove_punc=False)
        tokens = self.tokenize(cleaned_text)
        return tokens