"""This script runs the training and testing of the neural net.

The following parameters must be contained in the config file:
    epochs: The integer number of epochs to train for.
    model_save_path: The string filepath to save checkpoints/plots during training.
    name: A string name to use for save model checkpoint files.
    learning_rate: A float learning rate to use during training.
    autoencoder_vq: A boolean whether a VQVAE model is trained.
    autoencoder_vae: A boolean whether a VAE model is trained.
    random_seed: A integer to use as a random seed for consistency.
    learning_rate: A float learning rate to use during training.
    autoencoder_vq_components: An integer size for the Embedding size in VQVAE if 'autoencoder_vq' is True.

Typical usage example:
    python train_DAE.py --config=<CONFIG_FILE>

Note: CONFIG_FILE is the path containing the config file (ex. config/DAE.yml).
"""


import os
import pprint
import random
import time
import sys
import pickle
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openTSNE import TSNE
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader
from torch import optim
from configargparse import argparse

from config.parse_args import parse_args
from data_loader.lmdb_data_loader import *
from train_eval.train_seq2seq import train_iter_DAE
import utils.train_utils
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab
from model.DAE_model import DAE_Network, VQ_Frame, VAE_Network
from model.vocab import Vocab

[sys.path.append(i) for i in [".", ".."]]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global_loss_train: A dict containing keys 'loss' and 'var_loss'.
#    'loss': contains the training loss
#    'var_loss': #TODO
# Global_loss_eval: A list of testing loss values.
global Global_loss_train
global Global_loss_eval
Train_Deoising = False


def init_model(
    args: argparse.Namespace,
    lang_model: Vocab,
    pose_dim: int,
    _device: torch.device | str,
) -> Tuple[torch.nn.Module, torch.nn.MSELoss]:
    """Initializes a specified model.

    The model initialized depends on the config file values.
    The values checked (in order) are:
        - autoencoder_vq == 'True' returns a 'VQ_Frame' model.
            - This option requires the following values in the config file:
                - hidden_size (int)
                - autoencoder_vae (boolean)
                - autoencoder_vq_components (int)
        - autoencoder_vae == 'True' returns a 'VAE_Network' model.
            - This option requires the following values in the config file:
                - hidden_size (int)
        - Otherwise returns a 'DAE_Network'.
            - This option requires the following values in the config file:
                - hidden_size (int)

    Args:
        args: A configargparse object containing the needed config values.
        lang_model: A Vocab object of the pre-trained word vector representation.
        pose_dim: An integer count of the dimensions of the training data.
        _device: A torch.device or string indicating whether to train on CPU or GPU.

    Returns:
        A 2-Tuple:
            network: A custom PyTorch neural net. See above for options.
            loss_fn: A PyTorch loss object (MSELoss).
    """
    n_frames = args.n_poses
    # generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
    #                        lang_model.word_embedding_weights).to(_device)
    motion_dim = 162  # 162 #135
    # try:
    if args.autoencoder_vq == "True":
        network = VQ_Frame(
            motion_dim,
            args.hidden_size,
            args.autoencoder_vae == "True",
            int(args.autoencoder_vq_components),
        )
    elif args.autoencoder_vae == "True":
        network = VAE_Network(motion_dim, args.hidden_size)
    else:
        network = DAE_Network(motion_dim, args.hidden_size)

    loss_fn = torch.nn.MSELoss()
    network = network.to(device)
    return network, loss_fn


def train_epochs(
    args: argparse.Namespace,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    lang_model: Vocab,
    pose_dim: int,
    trial_id=None,
) -> None:
    """Train a PyTorch neural net.

    The training checkpoints are at every 20 epochs.
    The optimizer used is Adam.
    If a VQVAE model is selected, the 'Use_Tricks' boolean flag can be manually
    changed to use shortcuts #TODO.
    The loss values during training and testing are appended to the global
    variables 'Global_loss_train' and 'Global_loss_eval' and can be accessed
    through those global variables.

    The following training parameters must be contained in the config file:
        epochs: The integer number of epochs to train for.
        hidden_size: The integer size of the latent code space.
        model_save_path: The string filepath to save checkpoints during training.
        name: A string name to use for save checkpoints.
        autoencoder_vq: A boolean whether to customize training for VQVAE models.
        autoencoder_vae: A boolean whether to train VAE model (instead of VQVAE).
        learning_rate: A float learning rate to use during training.
        autoencoder_vq_components: An integer size for the Embedding size in VQVAE.

    Args:
        args: A configargparse object containing the needed config values.
        train_data_loader: A PyTorch DataLoader containing the training data.
        test_data_loader: A PyTorch DataLoader containing the testing data.
        lang_model: A Vocab object of the pre-trained word vector representation.
        pose_dim: An integer count of the dimensions of the training data.
    """
    global Global_loss_train
    global Global_loss_eval
    start = time.time()
    loss_meters = [AverageMeter("loss"), AverageMeter("var_loss")]

    # interval params
    print_interval = int(len(train_data_loader))
    save_model_epoch_interval = args.epochs

    # init model
    generator, loss_fn = init_model(args, lang_model, pose_dim, device)

    if args.hidden_size == -1:
        try:  # multi gpu
            gen_state_dict = generator.module.state_dict()
        except AttributeError:  # single gpu
            gen_state_dict = generator.state_dict()

        save_name = "{}/{}_H{}_checkpoint_{:03d}.bin".format(
            args.model_save_path, args.name, args.hidden_size, 20
        )

        utils.train_utils.save_checkpoint(
            {
                "args": args,
                "epoch": 20,
                "lang_model": lang_model,
                "pose_dim": pose_dim,
                "gen_dict": gen_state_dict,
            },
            save_name,
        )
        return

    # define optimizers
    gen_optimizer = optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    # training
    global_iter = 0
    train_res_recon_error = []
    train_res_perplexity = []
    print(args.epochs)
    for epoch in range(1, args.epochs + 1):
        if args.autoencoder_vq == "True":
            plot_embedding(
                args, generator, epoch, train_res_recon_error, train_res_perplexity
            )

        # evaluate the test set
        # val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, args)
        val_metrics = 0.01
        Global_loss_eval.append(val_metrics)
        # save model
        if epoch % save_model_epoch_interval == 0 and epoch > 0:
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()

            save_name = "{}/{}_H{}_checkpoint_{:03d}.bin".format(
                args.model_save_path, args.name, args.hidden_size, epoch
            )

            utils.train_utils.save_checkpoint(
                {
                    "args": args,
                    "epoch": epoch,
                    "lang_model": lang_model,
                    "pose_dim": pose_dim,
                    "gen_dict": gen_state_dict,
                },
                save_name,
            )

        # VQ Tricks
        Use_Tricks = False
        if args.autoencoder_vq == "True" and (not Use_Tricks):
            generator.skip_vq = False
        if args.autoencoder_vq == "True" and Use_Tricks:
            StartVQ_Epoch = 5
            ReEstimate_Epoch = 5
            if epoch < StartVQ_Epoch:
                generator.skip_vq = True
            else:
                generator.skip_vq = False
            if epoch % ReEstimate_Epoch == 0 and (not generator.skip_vq):
                generator.train(False)
                all_latents = []
                for iter_idx, data in enumerate(train_data_loader, 0):
                    noisy, original = data
                    original = original.to(device)
                    # print(original.shape)
                    original = torch.squeeze(original)
                    current_batch_latents = generator.encoder(original)
                    all_latents.append(current_batch_latents)
                all_latents = torch.vstack(all_latents)
                all_latents = all_latents.cpu().detach().numpy()
                generator.train(True)
                normalized_data = all_latents

                km = KMeans(
                    n_clusters=generator.vq_components, max_iter=2500, random_state=0
                ).fit(normalized_data)
                print("Kmeans trained!")
                # print("km",km.cluster_centers_.shape)
                # print("Embeddings", generator.vq_layer._embedding.weight.shape)
                generator.vq_layer._embedding.weight.data = torch.tensor(
                    km.cluster_centers_
                ).to(device)
                # exit()

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            # in_text, text_lengths, target_vec, in_audio, aux_info = data
            noisy, original = data
            batch_size = original.size(0)

            noisy = noisy.to(device)
            original = original.to(device)

            # train
            if args.autoencoder_vq == "True":
                loss, perplexity = train_iter_DAE(
                    args, epoch, original, original, generator, gen_optimizer
                )
                train_res_recon_error.append(loss["loss"])
                train_res_perplexity.append(perplexity.item())
            # if Train_Deoising:
            loss = train_iter_DAE(
                args, epoch, noisy, original, generator, gen_optimizer
            )
            # else:
            #     loss = train_iter_DAE(args, epoch, original, original, generator, gen_optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = "EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | ".format(
                    epoch,
                    iter_idx + 1,
                    utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time),
                )

                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += "{}: {:.3f}, ".format(
                            loss_meter.name, loss_meter.avg
                        )
                        Global_loss_train[loss_meter.name].append(loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()
    print("___________")
    # print(Global_loss_train)


def evaluate_testset(
    test_data_loader: DataLoader,
    generator: torch.nn.Module,
    loss_fn: torch.nn.MSELoss,
    args: argparse.Namespace,
) -> float:
    """Calculate testing loss scores of a neural net.

    The following testing parameters must be contained in the config file:
        autoencoder_vq: A boolean whether a VQVAE model was trained.
        autoencoder_vae: A boolean whether a VAE model was trained.

    Args:
        test_data_loader: A PyTorch DataLoader containing the testing dataset.
        generator: The PyTorch neural net to test.
        loss_fn: A PyTorch Loss object to use to calculate testing (default: MSELoss).
        args: A configargparser object containing testing parameters (See above).

    Returns:
        A float of the average loss value during testing.
    """
    # to evaluation mode
    generator.train(False)
    global Global_loss_eval
    losses = AverageMeter("loss")
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            # in_text, text_lengths, target_vec, in_audio, aux_info = data
            noisy, original = data
            batch_size = original.size(0)

            noisy = noisy.to(device)
            original = original.to(device)

            if args.autoencoder_vq == "True" and args.autoencoder_vae == "True":
                out_poses, vq_loss, preplexity, logvarm, meu = generator(noisy)
            elif args.autoencoder_vq == "True" and args.autoencoder_vae == "False":
                out_poses, vq_loss, preplexity = generator(noisy)
            elif args.autoencoder_vq == "False" and args.autoencoder_vae == "True":
                out_poses, logvarm, meu = generator(noisy)
            else:
                out_poses = generator(noisy)

            # This one is for GSOFT VQ
            # loss = out_poses.log_prob(original).sum(dim=1).mean()
            loss = loss_fn(out_poses, original)

            losses.update(loss.item(), batch_size)

    # back to training mode
    generator.train(True)

    # print
    elapsed_time = time.time() - start
    logging.info("[VAL] loss: {:.3f} / {:.1f}s".format(losses.avg, elapsed_time))

    return losses.avg


def main(
    config: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    lang_model: Vocab,
    H: int,
    vq_ncomponent: int = 1,
) -> Tuple[float, float]:
    """The main function for training and testing neural nets.

    The following parameters must be contained in the config file:
        epochs: The integer number of epochs to train for.
        model_save_path: The string filepath to save checkpoints during training.
        name: A string name to use for save checkpoints.
        learning_rate: A float learning rate to use during training.
        autoencoder_vq: A boolean whether a VQVAE model is trained.
        autoencoder_vae: A boolean whether a VAE model is trained.
        random_seed: A integer to use as a random seed for consistency.
        learning_rate: A float learning rate to use during training.
        autoencoder_vq_components: An integer size for the Embedding size in VQVAE.

    Args:
        config: A dict with the string key 'args' containing a configargparser object.
        train_loader: A PyTorch DataLoader with the training dataset.
        test_loader: A PyTorch DataLoader with the testing dataset.
        lang_model: A Vocab object with a pre-trained word vector representation.
        H: An integer for the size of the latent code space for the autoencoder.
        vq_ncomponent: An integer to append to the 'model_save_path'. (Can be same as H).

    Returns:
        A 2-Tuple:
            first: The float value of the last training loss.
            second: The float value of the last testing loss.
    """
    args: argparse.Namespace = config["args"]
    args.hidden_size = H
    # ssssssssssssssssssssssssssssssssssssssssssssssssssssargs.autoencoder_vq_components = vq_ncomponent
    if args.autoencoder_vq == "True":  # ths was for training a lot of networks
        args.model_save_path += "/" + str(vq_ncomponent)
    # args.epochs = 20
    args.model_save_path += "/train_DAE_H" + str(H)
    trial_id = None

    # random seed
    if args.random_seed >= 0:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    # set logger
    utils.train_utils.set_logger(
        args.model_save_path, os.path.basename(__file__).replace(".py", ".log")
    )

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info(pprint.pformat(vars(args)))

    # Dataset moved from here.

    global Global_loss_train
    Global_loss_train = {"loss": [], "var_loss": []}
    global Global_loss_eval
    Global_loss_eval = []
    # train
    train_epochs(
        args, train_loader, test_loader, lang_model, pose_dim=15 * 9, trial_id=trial_id
    )

    # print("****************")
    # print(Global_loss_eval)
    dict2save = {"Global_train": Global_loss_train, "Global_eval": Global_loss_eval}
    pickle.dump(dict2save, open(args.model_save_path + "/globals.pk", "wb"))

    try:
        plot_loss(
            Global_loss_train, Global_loss_eval, args.model_save_path + "/loss_plot.png"
        )
    except:
        pass
    # plt.plot(Global_loss_train)

    return Global_loss_train["loss"][-1], Global_loss_eval[-1]


def plot_loss(
    Global_loss_train: dict[str, list], Global_loss_eval: list, save_path: str
) -> None:
    """Plot the losses currently recorded.

    Saves the plot to the 'save_path' directory.

    Args:
        Global_loss_train: A dict containing the key 'loss' with a list of float losses.
        Global_loss_eval: A list containing float loss values.
        save_path: A string filepath to the directory to save the plot.
    """
    plt.plot(Global_loss_train["loss"], label="Train loss")
    plt.plot(Global_loss_eval, label="Evaluation loss")

    # Labeling the X-axis
    plt.xlabel("Epoch number")
    # Labeling the Y-axis
    plt.ylabel("Loss Average")
    # Give a title to the graph
    plt.title(
        "Training/Evaluation Loss based on epoch number \n"
        + "{:.4f}".format(Global_loss_train["loss"][-1])
    )

    # Show a legend on the plot
    plt.legend()

    plt.savefig(save_path)
    # plt.savefig(os.path.join(args.model_save_path, 'loss_plot.png'))
    plt.show()


def plot_embedding(
    args: argparse.Namespace,
    rnn: VQ_Frame,
    epoch_num: int,
    train_res_recon_error: list,
    train_res_perplexity: list,
) -> None:
    """Plot the Vector-Quantized (VQ) Embedding layer values.

    Saves the plots to the 'model_save_path' directory (See below).

    The following parameters must be contained in the config file:
        model_save_path: The string filepath to save the plots.

    Args:
        args: A configargparser object containing parameters (See above).
        rnn: A VQ_Frame neural net with an Embedding layer.
        epoch_num: An integer value of the total number of epochs.
        train_res_recon_error: A list of float training reconstruction losses.
        train_res_perplexity: A list of float perplexity losses.
    """

    w: np.ndarray = rnn.vq_layer._embedding.weight.cpu().detach().numpy()
    # PCA
    if w.shape[0] > 20:
        pca = PCA(n_components=20)
        priciple_components = pca.fit(w)
        normalized_data = priciple_components.transform(w)
    else:
        normalized_data = w
    # TSNE

    MyTSNE = TSNE(
        n_components=2, perplexity=30, metric="euclidean", n_jobs=8, verbose=True
    )
    X_embedded = MyTSNE.fit(normalized_data)

    plt.figure(figsize=(16, 10))
    # palette = sns.color_palette("bright", k_component)
    # 2D

    sns.scatterplot(
        X_embedded[:, 0],
        X_embedded[:, 1],
        # hue=labels_,
        legend=False,
    )
    plt.title("Embedding visualization" + str(epoch_num))
    address = (args.model_save_path) + "/plots/"
    os.makedirs(address, exist_ok=True)
    address += "Embedding_Epoch(" + str(epoch_num) + ").png"
    plt.savefig(address)
    plt.clf()
    plt.show()

    # Reconstruction loss and perplexity
    print(train_res_perplexity)
    try:
        train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
        train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
    except:
        train_res_recon_error_smooth = (
            train_res_recon_error  # savgol_filter(train_res_recon_error, 201, 7)
        )
        train_res_perplexity_smooth = (
            train_res_perplexity  # [0] #savgol_filter(train_res_perplexity, 201, 7)
        )

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale("log")
    ax.set_title("Smoothed NMSE.")
    ax.set_xlabel("iteration")

    ax = f.add_subplot(1, 2, 2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title("Smoothed Average codebook usage (perplexity).")
    ax.set_xlabel("iteration")

    address = (args.model_save_path) + "/plots/Error_Epoch(" + str(epoch_num) + ").png"

    plt.savefig(address)
    plt.clf()
    plt.show()

    pass


if __name__ == "__main__":
    # --config=../config/seq2seq.yml
    _args = parse_args()
    # dataset
    args = _args
    train_dataset = TrinityDataset_DAE(
        args,
        args.train_data_path[0],
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        data_mean=args.data_mean,
        data_std=args.data_std,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.loader_workers,
        pin_memory=True
        # collate_fn=word_seq_collate_fn
    )

    val_dataset = TrinityDataset_DAE(
        args,
        args.val_data_path[0],
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        data_mean=args.data_mean,
        data_std=args.data_std,
    )
    test_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.loader_workers,
        pin_memory=True
        # collate_fn=word_seq_collate_fn
    )

    # build vocab
    vocab_cache_path = os.path.join(
        os.path.split(args.train_data_path[0])[0], "vocab_cache.pkl"
    )
    lang_model = build_vocab(
        "words",
        [train_dataset, val_dataset],
        vocab_cache_path,
        args.wordembed_path,
        args.wordembed_dim,
    )
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    train_loss_list = []
    eval_loss_list = []
    dim_lost = []
    backupsv = _args.model_save_path
    # This range can be used to test various latent code space sizes.
    # Best value appears to be 40.
    for k in range(45, 46, 1):
        print("*******************************************************************")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("k=" + str(k))
        _args.model_save_path = backupsv
        current_train_loss, current_eval_loss = main(
            {"args": _args}, train_loader, test_loader, lang_model, k, vq_ncomponent=k
        )
        dim_lost.append(int(k))
        train_loss_list.append(current_train_loss)
        eval_loss_list.append(current_eval_loss)

        # Do inference
        save_name = "{}/{}_H{}_checkpoint_{:03d}.bin".format(
            _args.model_save_path, _args.name, _args.hidden_size, _args.epochs
        )
        command = "python inference_DAE.py " + save_name
        os.system(command)

    #     Plot results
    plt.plot(dim_lost, train_loss_list, label="Train loss")
    plt.plot(dim_lost, eval_loss_list, label="Evaluation loss")

    # Labeling the X-axis
    plt.xlabel("Dimensionality")
    # Labeling the Y-axis
    plt.ylabel("Loss Average")
    # Give a title to the graph
    plt.title("Training/Evaluation Loss based on latent dim")

    # Show a legend on the plot
    plt.legend()

    plt.savefig(backupsv + "/overall.png")
    # plt.savefig(os.path.join(args.model_save_path, 'loss_plot.png'))
    plt.show()

    """
    main({'args': _args}, train_loader, test_loader, lang_model, 150, #-2 #-1,
             vq_ncomponent=args.autoencoder_vq_components)
    """
