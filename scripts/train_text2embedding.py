"""Part c: Gestures Sequence Chunking and Part d: Text to Gesture Translation
"""


from __future__ import annotations
import os
import pprint
import random
import time
import sys
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from configargparse import argparse

[sys.path.append(i) for i in [".", ".."]]

from config.parse_args import parse_args
from data_loader.lmdb_data_loader import *
from model.vocab import Vocab
from model.seq2seq_net import Seq2SeqNet
from model.text2embedding_model import text2embedding_model, text2embedding_model_New
from train_eval.train_seq2seq import (
    train_iter_seq2seq,
    train_iter_c2g_seq2seq,
    train_iter_text2embedding,
)
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab
import utils.train_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
debug = False


def init_model(
    args, lang_model: Vocab, pose_dim: int, _device: torch.device | None
) -> Tuple[text2embedding_model, torch.nn.MSELoss]:
    """ """
    n_frames: int = args.n_poses
    try:
        if args.text2_embedding_discrete == "True":
            pose_dim = int(args.autoencoder_vq_components)
    except:
        pass
    generator = text2embedding_model(
        args,
        pose_dim,
        n_frames,
        lang_model.n_words,
        args.wordembed_dim,
        lang_model.word_embedding_weights,
    ).to(_device)
    loss_fn = torch.nn.MSELoss()

    return generator, loss_fn


def train_epochs(
    args: argparse.Namespace,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    lang_model: Vocab,
    pose_dim: int,
    trial_id=None,
) -> None:
    """ """
    start = time.time()
    loss_meters = [AverageMeter("loss"), AverageMeter("var_loss")]

    # interval params
    print_interval = np.max((int(len(train_data_loader) / 5), 1))
    save_model_epoch_interval = 10

    # init model
    generator, loss_fn = init_model(args, lang_model, pose_dim, device)
    args_backup = args
    start_epcoh = 1
    Check_point_epoch = 0
    load_pre_trained = False
    if load_pre_trained:
        Check_point_epoch = 250
        addr = "{}/{}_checkpoint_{:03d}.bin".format(
            args.model_save_path, args.name, Check_point_epoch
        )
        txt2embedding_model = utils.train_utils.load_checkpoint_and_model(
            addr, device, what="text2embedding"
        )
        args, generator, loss_fn, lang_model, pose_dim = txt2embedding_model
        start_epcoh = Check_point_epoch
    # define optimizers
    gen_optimizer = optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    args.epochs = args_backup.epochs

    # training
    global_iter = 0

    # To record loss and evaluations metric:
    val_metrics_list = []
    loss_list = []
    perplexity_list = []

    for epoch in range(start_epcoh, Check_point_epoch + args.epochs + 1):
        # evaluate the test set
        val_metrics = 0
        val_metrics, perplexity = evaluate_testset(
            test_data_loader, generator, loss_fn, args
        )
        val_metrics_list.append(val_metrics)
        perplexity_list.append(perplexity)
        # break
        # save model
        if epoch % save_model_epoch_interval == 0 and epoch > 0:
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()

            save_name = "{}/{}_checkpoint_{:03d}.bin".format(
                args.model_save_path, args.name, epoch
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

        # train iter
        iter_start_time = time.time()

        loss_epoch = AverageMeter("loss")
        loss_epoch.reset()

        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1

            (
                in_text,
                text_lengths,
                target_vec,
                in_audio,
                aux_info,
                sentence_leve_latents,
                c_portion,
                GPT3_Embedding,
            ) = data
            batch_size = target_vec.size(0)

            # print('intext', in_text, '\nCindec', c_portion )

            in_text = in_text.to(device)
            in_audio = in_audio.to(device)
            # target_vec = target_vec.to(device)
            target_vec = sentence_leve_latents.to(device)
            c_portion = c_portion.to(device)
            GPT3_Embedding = GPT3_Embedding.to(device)
            # train
            loss = train_iter_text2embedding(
                args,
                epoch,
                in_text,
                text_lengths,
                in_audio,
                target_vec,
                c_portion,
                GPT3_Embedding,
                generator,
                gen_optimizer,
            )

            # loss values
            loss_epoch.update(loss["loss"], batch_size)
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
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()
        loss_list.append(loss_epoch.avg)

    plot_loss(
        args,
        all_eval_loss=val_metrics_list,
        all_train_loss=loss_list,
        perplexities=(perplexity_list),
    )


def evaluate_testset(
    test_data_loader: DataLoader,
    generator: text2embedding_model,
    loss_fn: torch.nn.MSELoss,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    """ """
    # to evaluation mode
    generator.train(False)

    losses = AverageMeter("loss")
    start = time.time()

    perplexities = AverageMeter("perplexity")

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            (
                in_text,
                text_lengths,
                target_vec,
                in_audio,
                aux_info,
                sentence_leve_latents,
                cluster_ids,
                GPT3_embeddings,
            ) = data

            batch_size = target_vec.size(0)

            if debug:
                print(
                    "sentence_level_latents.shape", sentence_leve_latents.shape
                )  # [128, 4, 400]
                print("cluster_ids", cluster_ids.shape)
            in_text = in_text.to(device)
            in_audio = in_audio.to(device)
            # target = target_vec.to(device)
            target = sentence_leve_latents.to(device)
            cluster_ids = cluster_ids.to(device)
            GPT3_embeddings = GPT3_embeddings.to(device)
            if generator.text2_embedding_discrete:
                out_latents, _ = generator(
                    in_text, text_lengths, in_audio, cluster_ids, GPT3_embeddings, None
                )
            else:
                out_latents, _ = generator(
                    in_text, text_lengths, in_audio, target, None
                )

            if args.text2_embedding_discrete == "False":
                if debug:
                    print("!", out_latents.shape, out_latents)
                loss = loss_fn(out_latents, target)
                losses.update(loss.item(), batch_size)
            else:
                # 1. CrossEntropy Loss

                cluster_targets_one_hot = F.one_hot(
                    cluster_ids.reshape(-1).to(torch.int64), 514
                )

                # cluster_targets_one_hot = cluster_targets_one_hot.reshape(os[0], os[1], -1)
                if debug:
                    print("check shape before reshape", out_latents.shape)
                out_latents = out_latents.reshape(-1, out_latents.shape[2])

                if debug:
                    print(
                        "check shape",
                        out_latents.shape,
                        cluster_targets_one_hot.shape,
                        cluster_ids.reshape(-1).shape,
                    )
                # loss = torch.nn.MSELoss()(out_latents, cluster_targets_one_hot)
                loss = torch.nn.CrossEntropyLoss()(
                    out_latents, cluster_ids.reshape(-1).to(torch.long)
                )
                losses.update(loss.item(), batch_size)

                # 2. Perplexity
                encoding_indices = out_latents.argmax(1).float().unsqueeze(1)

                encodings = torch.zeros(
                    encoding_indices.shape[0], 514, device=encoding_indices.device
                )
                encodings.scatter_(1, encoding_indices.to(torch.int64), 1)
                avg_probs = torch.mean(encodings, dim=0)
                perplexity = torch.exp(
                    -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
                )
                # perplexity = torch.nn.CrossEntropyLoss()(out_latents.float(),
                #                                          cluster_ids.view(-1).long())
                perplexity = perplexity.detach().cpu().numpy()
                # print(out_latents.shape, avg_probs, perplexity)
                perplexities.update(perplexity, batch_size)

                # 3. BLEU Score

    # back to training mode
    generator.train(True)

    # print
    elapsed_time = time.time() - start
    logging.info("[VAL] loss: {:.3f} / {:.1f}s".format(losses.avg, elapsed_time))

    return losses.avg, perplexities.avg


def plot_loss(
    args: argparse.Namespace,
    all_eval_loss: list[float],
    all_train_loss: list[float],
    perplexities: list[float],
) -> None:
    # X = np.arange(136-3)
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.bar(X + 0.00, all_train_loss[0], color='b', width=0.25)
    # ax.bar(X + 0.25, all_eval_loss[1], color='g', width=0.25)
    # plt.show()
    # plotting the second plot
    plt.plot(all_train_loss, label="Train loss")
    plt.plot(all_eval_loss, label="Evaluation loss")

    # Labeling the X-axis
    plt.xlabel("Epoch number")
    # Labeling the Y-axis
    plt.ylabel("Loss Average")
    # Give a title to the graph
    plt.title("Training/Evaluation Loss based on epoch number")

    # Show a legend on the plot
    plt.legend()

    plt.savefig(os.path.join(args.model_save_path, "loss_plot.png"))
    plt.show()

    plt.title("perplexities=" + str(perplexities[-1]))
    plt.plot(perplexities, label="aa")
    plt.savefig(os.path.join(args.model_save_path, "Perplexity.png"))
    plt.show()


def main(config: dict) -> None:
    args: argparse.Namespace = config["args"]

    args.model_save_path = (
        os.path.dirname(args.autoencoder_checkpoint) + "/text2mbedding/"
    )

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

    # dataset
    train_dataset = TrinityDataset_sentencelevel(
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
        pin_memory=True,
        collate_fn=word_seq_collate_fn,
    )

    val_dataset = TrinityDataset_sentencelevel(
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
        pin_memory=True,
        collate_fn=word_seq_collate_fn,
    )
    print("Data loaded successfully")
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

    # train
    train_epochs(
        args, train_loader, test_loader, lang_model, pose_dim=15 * 9, trial_id=trial_id
    )


if __name__ == "__main__":
    _args = parse_args()
    main({"args": _args})
