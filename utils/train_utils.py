import logging
import os
import random
from logging.handlers import RotatingFileHandler

import numpy as np
import time
import math
import torch

from train_DAE import init_model as DAE_init
from train_Autoencoder import init_model as autoencoder_init
from train_cluster2gesture import init_model as c2g_init
from train_text2embedding import init_model as text2embedding_init
from train_gan import init_model as text2embedding_gan_init
from train_autoencoder_VQVAE import init_model as VQVAE_init
from train import init_model as baseline_init_model
def set_logger(log_path=None, log_filename='log'):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        handlers.append(
            RotatingFileHandler(os.path.join(log_path, log_filename), maxBytes=10 * 1024 * 1024, backupCount=5))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since):
    now = time.time()
    s = now - since
    return '%s' % as_minutes(s)


def save_checkpoint(state, filename):
    torch.save(state, filename)
    logging.info('Saved the checkpoint')


def load_checkpoint_and_model(checkpoint_path, _device='cpu', what=''):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    if what == 'c2g':
        generator, loss_fn = c2g_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint['gen_dict'])
    elif what == 'text2embedding':
        generator, loss_fn = text2embedding_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint['gen_dict'])
        # Todo: should change to what==? and then fix all it's callings.
    elif what == 'DAE': #Todo
        generator, loss_fn = DAE_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint['gen_dict'])
    elif what == "autoencoder": #Todo
        generator, loss_fn = autoencoder_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint['gen_dict'])
    elif what == 'baseline':
        generator, loss_fn = baseline_init_model(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint['gen_dict'])
    elif what == 'text2embedding_gan':
        generator, loss_fn = text2embedding_gan_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint['gen_dict'])
    elif what == 'autoencoder_vq':
        generator, loss_fn = VQVAE_init(args, lang_model, pose_dim, _device)
        generator.load_state_dict(checkpoint['gen_dict'])
    else:
        assert (1==2)


    # set to eval mode
    generator.train(False)

    return args, generator, loss_fn, lang_model, pose_dim
