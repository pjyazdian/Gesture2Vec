"""Utility file to load and save model training checkpoints.

Saves the model and associated parameters into a .bin file.
Typically used to save model progress after every 20%.
If loading a model, must specify the model string.
Options: 'c2g', 'text2embedding', 'DAE', 'autoencoder', 'baseline', 'text2embedding_gan', 'autoencoder_vq'

Typical usage example:
    save_checkpoint({
        args: ArgumentParser object with the current parameters.
        epoch: The epoch that have been completed.
        lang_model: Vocab object containing the trained word vector representation.
        pose_dim: An integer value of the number of dimensions of a single gesture.
        gen_dict: A state_dict from a PyTorch Neural Net subclass.args:
    }, 'autoencoder_progress_20.bin')

    or

    m = load_checkpoint_and_model('output/model.bin', 'gpu', 'autoencoder_vq')
"""


import logging
import os
from logging.handlers import RotatingFileHandler

import time
import math
from typing import Tuple
from configargparse import argparse
import torch

from model.vocab import Vocab
from train_DAE import init_model as DAE_init
from train_Autoencoder import init_model as autoencoder_init
from train_cluster2gesture import init_model as c2g_init
from train_text2embedding import init_model as text2embedding_init
from train_gan import init_model as text2embedding_gan_init
from train_autoencoder_VQVAE import init_model as VQVAE_init
from train import init_model as baseline_init_model


def set_logger(log_path: str = None, log_filename: str = "log") -> None:
    """Set the logger with a given log name and directory.

    Max filesize limit is 10MB with up to 5 logs by default.

    Args:
        log_path: The string specifying the directory to save logs.
        log_filename: The string specifying the name of the log.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                os.path.join(log_path, log_filename),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
            )
        )
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s: %(message)s", handlers=handlers
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def as_minutes(s: int) -> str:
    """Convert seconds into a mins, seconds string.

    Args:
        s: The seconds as an integer.

    Returns:
        A string in the format of '<minutes> <seconds>'.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since: int) -> str:
    """Calculate the time between now and a specific time in seconds.

    Args:
        since: The starting time in seconds.

    Returns:
        A string showing the elapsed time in the format of 'minutes seconds'.
    """
    now = time.time()
    s = now - since
    return "%s" % as_minutes(s)


def save_checkpoint(state: dict, filename: str) -> None:
    """Save the current model into a bin file.

    Args:
        state: A dictionary.
            {
                args: ArgumentParser object with the current parameters.
                epoch: The epoch that have been completed.
                lang_model: Vocab object containing the trained word vector representation.
                pose_dim: An integer value of the number of dimensions of a single gesture.
                gen_dict: A state_dict from a PyTorch Neural Net subclass.
            }
        filename: The filename to save the current model into.
    """
    torch.save(state, filename)
    logging.info("Saved the checkpoint")


def load_checkpoint_and_model(
    checkpoint_path, _device = "cpu", what: str = ""
):
    """Load a checkpoint file representing a saved state of a model into memory.

    Args:
        checkpoint_path: The string filepath to find the file to load.
        _device: A string or torch.device indicating the availability of a GPU. Default value is 'cpu'.
        what: A string specifying the particular model to load.
            Options: 'c2g', 'text2embedding', 'DAE', 'autoencoder', 'baseline', 'text2embedding_gan', 'autoencoder_vq'

    Returns a Tuple:
        args: ArgumentParser object containing model and data parameters used.
        generator: The model loaded.
        loss_fn: A PyTorch loss function used to score the model.
        lang_model: A Vocab object that contains the pre-trained word vector representations used.
        pose_dim: An integer value of the number of dimensions of a gesture.

    Raises:
        An assertion if the 'what' arg specifies a non-existing model.
    """
    print("loading checkpoint {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args: argparse.Namespace = checkpoint["args"]
    epoch: int = checkpoint["epoch"]
    lang_model: Vocab = checkpoint["lang_model"]
    pose_dim: int = checkpoint["pose_dim"]
    print("epoch {}".format(epoch))

    if what == "c2g":
        generator, loss_fn = c2g_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint["gen_dict"])
    elif what == "text2embedding":
        generator, loss_fn = text2embedding_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint["gen_dict"])
        # Todo: should change to what==? and then fix all it's callings.
    elif what == "DAE":  # Todo
        generator, loss_fn = DAE_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint["gen_dict"])
    elif what == "autoencoder":  # Todo
        generator, loss_fn = autoencoder_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint["gen_dict"])
    elif what == "baseline":
        generator, loss_fn = baseline_init_model(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint["gen_dict"])
    elif what == "text2embedding_gan":
        generator, loss_fn = text2embedding_gan_init(
            args, lang_model, pose_dim, _device
        )
        generator.load_state_dict(checkpoint["gen_dict"])
    elif what == "autoencoder_vq":
        generator, loss_fn = VQVAE_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint["gen_dict"])
    else:
        assert 1 == 2

    # set to eval mode
    generator.train(False)

    return args, generator, loss_fn, lang_model, pose_dim
