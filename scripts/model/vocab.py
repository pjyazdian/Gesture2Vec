"""This class represents the vocabulary of a language.

This class uses the FastText library and pre-trained data to analyze text.
The pre-trained data must be provided and loaded before other methods are used.

Typical usage example:
    v = Vocab('first_vocab)
    v.load_word_vectors('resource/crawl_300d_2M-subword.bin')
    for i in range(list_of_words):
        v.index_word(i)
"""


import logging
import os
import pickle
import numpy as np
import fasttext


class Vocab:
    """Contains the vocabulary of a language

    This class contains word vector representations (in the English language) using FastText pre-trained data.
    Default tokens are:
        PAD (Padding) - Token for padding (to account for different length sentences).
        SOS (Start of Sentence) - Token for signifying the start of a sentence.
        EOS (End of Sentence) - Token for signifying the end of a sentence.
        UNK (Unknown) - Token for words that have not been previously encountered.

    Attributes:
        name: A string that gives a custom name to this object
        trimmed: A boolean that indicates whether this vocabulary has removed words that fall below a (custom) minimum count threshold.
        word_embedding_weights: A Numpy array representation of a word.
        word2index: A string,integer dictionary that contains all words that exist in the vocabulary and the index of each word within the word vector representation.
        word2count: A string,integer dictionary that contains all words that exist in the vocabulary and the count of each word previously encountered.
        index2word: An integer,string dictionary that contains all indices that exist in the vocabulary and each word found at the index.
        n_words: An int representing the number of words in this vocabulary.
    """

    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3

    def __init__(self, name: str, insert_default_tokens: bool = True):
        """Initialization function.

        Args:
            name: A string giving this object a printable name.
            insert_default_tokens: A boolean that is used as the argument for the reset_dictionary() class function below.
        """
        self.name = name
        self.trimmed = False
        self.word_embedding_weights = None
        self.reset_dictionary(insert_default_tokens)

    def reset_dictionary(self, insert_default_tokens: bool = True) -> None:
        """Reset the dictionary (tokens) of the language.

        Remove all existing tokens except for certain base tokens.
        Modifies the internal state of this object.

        The base tokens are as follows:
        PAD (Padding) - Token for padding (to account for different length sentences).
        SOS (Start of Sentence) - Token for signifying the start of a sentence.
        EOS (End of Sentence) - Token for signifying the end of a sentence.
        UNK (Unknown) - Token for words that have not been previously encountered.

        Default behavior is to include all base tokens (padding and sentence tokens).

        Args:
            insert_default_tokens: A boolean to signify whether PAD/SOS/EOS tokens should be included after resetting the dictionary.
        """
        self.word2index = {}
        self.word2count = {}
        if insert_default_tokens:
            self.index2word = {
                self.PAD_token: "<PAD>",
                self.SOS_token: "<SOS>",
                self.EOS_token: "<EOS>",
                self.UNK_token: "<UNK>",
            }
        else:
            self.index2word = {self.UNK_token: "<UNK>"}
        self.n_words = len(self.index2word)  # count default tokens

    def index_word(self, word: str) -> None:
        """Increments the count of a specified word.

        Keep track of the count of every word encountered by incrementing the word count of the specified word.
        If a word has not been seen before, then create a new entry. Modifies the internal state of this object.

        Args:
            word: A string that represents the word and acts as the index.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_vocab(self, other_vocab: "Vocab") -> None:
        """Adds word(s) to the vocabulary by concatenating an existing vocabulary.

        Concatenates two vocabularies together.
        Modifies internal state of the calling object.
        Does not affect the internal state of the provided object.

        Args:
            other_vocab: A Vocab object containing the desired vocabulary to merge.
        """
        for word, _ in other_vocab.word2count.items():
            self.index_word(word)

    def trim(self, min_count: int) -> None:
        """Remove words below a certain count threshold.

        Remove any word in the vocabulary that does not meet a minimum count threshold.
        All words that do not meet the minimum are removed.
        Modifies the internal state of the object.

        Args:
            min_count: An integer for the minimum count threshold to not be removed from the vocabulary.
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        logging.info(
            "    word trimming, kept %s / %s = %.4f"
            % (
                len(keep_words),
                len(self.word2index),
                len(keep_words) / len(self.word2index),
            )
        )

        # reinitialize dictionary
        self.reset_dictionary()
        for word in keep_words:
            self.index_word(word)

    def get_word_index(self, word: str) -> int:
        """Get the index of a word in the vector representation.

        Args:
            word: A string for the particular word that index should be retrieved.

        Returns:
            The index of the word or the index of the <Unknown> token if the word has not been seen previously.
        """
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.UNK_token

    def load_word_vectors(self, pretrained_path: str, embedding_dim: int = 300) -> None:
        """Load an existing (FastText) word vector model.

        Requires a pre-existing .bin file to be available.
        Nominally expected in the 'PROJECT_ROOT/resource folder'.
        Modifies internal state of the object.

        Args:
            pretrained_path: A string filepath to the FastText model.
            embedding_dim: An integer count of the dimensions in the FastText model (default 300).
        """
        logging.info("  loading word vectors from '{}'...".format(pretrained_path))

        # initialize embeddings to random values for special words
        init_sd = 1 / np.sqrt(embedding_dim)
        weights = np.random.normal(0, scale=init_sd, size=[self.n_words, embedding_dim])
        weights = weights.astype(np.float32)

        # read word vectors
        word_model = fasttext.load_model(pretrained_path)
        for word, id in self.word2index.items():
            vec = word_model.get_word_vector(word)
            weights[id] = vec

        self.word_embedding_weights = weights

    def __get_embedding_weight(
        self, pretrained_path: str, embedding_dim: int = 300
    ) -> np.ndarray | None:
        """Returns embedding weights.

        Function modified from http://ronny.rest/blog/post_2017_08_04_glove/
        If weights have not been calculated then derive using the data in the pretrained_path.
        Saves a pickle file using the 'cache_path' attribute.
        If a cached pickle file already exists, then use the cache directly.

        Args:
            pretrained_path: The string filepath of pretrained word representation vectors.
            embedding_dim: An integer count of the dimensions in the FastText model (default 300).

        Returns:
            A (float) Numpy array of the embedding weights.
            None if a cache version already exists but does not match the embedding_dim.
        """
        logging.info("Loading word embedding '{}'...".format(pretrained_path))
        cache_path = os.path.splitext(pretrained_path)[0] + "_cache.pkl"
        weights = None

        # use cached file if it exists
        if os.path.exists(cache_path):  #
            with open(cache_path, "rb") as f:
                logging.info("  using cached result from {}".format(cache_path))
                weights = pickle.load(f)
                if weights.shape != (self.n_words, embedding_dim):
                    logging.warning(
                        "  failed to load word embedding weights. reinitializing..."
                    )
                    weights = None

        if weights is None:
            # initialize embeddings to random values for special and OOV words
            init_sd = 1 / np.sqrt(embedding_dim)
            weights = np.random.normal(
                0, scale=init_sd, size=[self.n_words, embedding_dim]
            )
            weights = weights.astype(np.float32)

            with open(pretrained_path, encoding="utf-8", mode="r") as textFile:
                num_embedded_words = 0
                for line_raw in textFile:
                    # extract the word, and embeddings vector
                    line = line_raw.split()
                    try:
                        word, vector = (line[0], np.array(line[1:], dtype=np.float32))
                        # if word == 'love':  # debugging
                        #     print(word, vector)

                        # if it is in our vocab, then update the corresponding weights
                        id = self.word2index.get(word, None)
                        if id is not None:
                            weights[id] = vector
                            num_embedded_words += 1
                    except ValueError:
                        logging.info("  parsing error at {}...".format(line_raw[:50]))
                        continue
                logging.info(
                    "  {} / {} word vectors are found in the embedding".format(
                        num_embedded_words, len(self.word2index)
                    )
                )

                with open(cache_path, "wb") as f:
                    pickle.dump(weights, f)

        return weights
