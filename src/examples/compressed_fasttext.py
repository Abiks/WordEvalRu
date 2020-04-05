import compress_fasttext
from nltk.tokenize import WordPunctTokenizer
import numpy as np

DEFAULT_MODEL = 'https://github.com/avidale/compress-fasttext/releases/download/v0.0.1/ft_freqprune_100K_20K_pq_100.bin'


class CompressedFastTextEmbedder:
    def __init__(self, path_to_model=None):
        """
        Initialize model. If path_to_model is None initialize model from:
        https://github.com/avidale/compress-fasttext/releases/download/v0.0.1/ft_freqprune_100K_20K_pq_100.bin
        """
        if path_to_model is None:
            path_to_model = DEFAULT_MODEL
           
        self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(path_to_model)

    def get_word_embeddings(self, sentence):
        """
        If sentence is just text tokenize the passed sentence using WordPunctTokenizer from nltk package
        Then construct matrix of embeddings of all tokens. 
        
        return numpy array of shape (number_of_tokens, len_of_word_embedding)         
        """
        if isinstance(sentence, str):
            sentence_tokenized = WordPunctTokenizer().tokenize(sentence)
        elif isinstance(sentence, list) and all(isinstance(token, str) for token in sentence):
            sentence_tokenized = sentence
        else:
            raise TypeError

        embeddings = []

        for token in sentence_tokenized:
            embeddings.append(self.model[token.lower()])
        return np.vstack(embeddings)
