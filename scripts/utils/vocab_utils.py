"""Utility file to build a a language vector representation model.

Language vector representations are built from existing FastText resources.
These resource files (ex. crawl-300d-2M-subword.bin) must be saved to a directory.
The directory must be included in the config files.
Training and Testing datasets (as PyTorch dataset subclasses) must also be provided.
If a cache file is valid, then the word_vec_path and feat_dim args can be ignored.

Typical usage example:
    v = build_vocab(
        'first_vocab',
        [training, testing],
        'dataset/vocab_cache.pkl',
        'resource/crawl-300d-2M-subword.bin',
        300
    )
    index_words(v, 'dataset/lmdb/data.mdb')
"""


import logging
import os
import pickle

import lmdb
import pyarrow

from model.vocab import Vocab


def build_vocab(
    name: str,
    dataset_list: list,
    cache_path: str,
    word_vec_path: str = None,
    feat_dim = None,
) -> Vocab:
    """Build a language vector representation model from an existing source.

    Builds a language model using existing (English) FastText vector representations.
    The 'word_vec_path' and 'feat_dim' arguments must be provided if a model has not been previously created.
    Once the model has been built, saves the model using Pickle to the 'cache_path' location.
    If an existing model has been detected at the 'cache_path' then load the model instead of build.

    Args:
        name: A string to be used as a name for the language model.
        dataset_list: A list containing PyTorch 'Dataset' objects that are represented within Lmdb files and contains dataset associated information.
        cache_path: A string representing the filepath to check if a language model has been previously built.
        word_vec_path: A string representing (FastText) .bin files to use.
        feat_dim: An int representing the dimensions in the FastText files.

    Returns:
        A Vocab object that contains the language vector representations.

    Raises:
        Assertion that the model is consistent with its embedded weights.
    """
    logging.info("  building a language model...")
    if not os.path.exists(cache_path):
        lang_model = Vocab(name)
        for dataset in dataset_list:
            logging.info("    indexing words from {}".format(dataset.lmdb_dir))
            index_words(lang_model, dataset.lmdb_dir)

        if word_vec_path is not None:
            lang_model.load_word_vectors(word_vec_path, feat_dim)

        with open(cache_path, "wb") as f:
            pickle.dump(lang_model, f)
    else:
        logging.info("    loaded from {}".format(cache_path))
        with open(cache_path, "rb") as f:
            lang_model: Vocab = pickle.load(f)

        if word_vec_path is None:
            lang_model.word_embedding_weights = None
        elif lang_model.word_embedding_weights.shape[0] != lang_model.n_words:
            logging.warning("    failed to load word embedding weights. check this")
            assert False

    return lang_model


def index_words(lang_model: Vocab, lmdb_dir: str) -> None:
    """Analyzes and indexes the words in the dataset to a Vocab object.

    Modifies the lang_model object by calling a mutating method.
    Adds all words in the lmdb file to the lang_model.

    Args:
        lang_model: A Vocab object representing the language model to train.
        lmdb_dir: A string representing the filepath of the dataset to analyze.
    """
    lmdb_env: lmdb.Environment = lmdb.open(lmdb_dir, readonly=True, lock=False)
    txn = lmdb_env.begin(write=False)
    cursor = txn.cursor()

    for key, buf in cursor:
        video = pyarrow.deserialize(buf)

        for clip in video["clips"]:
            for word_info in clip["words"]:
                word = word_info[0]
                lang_model.index_word(word)

    lmdb_env.close()
    logging.info("    indexed %d words" % lang_model.n_words)
