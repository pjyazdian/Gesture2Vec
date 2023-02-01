import datetime
import os
import pprint
import random
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
[sys.path.append(i) for i in ['.', '..']]

from model import vocab
from model.seq2seq_net import Seq2SeqNet
from model.Autoencoder_VQVAE_model import Autoencoder_VQVAE
from train_eval.train_seq2seq import train_iter_seq2seq, train_iter_Autoencoder_seq2seq, train_iter_Autoencoder_ssl_seq2seq, train_iter_Autoencoder_VQ_seq2seq
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab

from config.parse_args import parse_args

from torch import optim

from data_loader.lmdb_data_loader import *
import utils.train_utils

import utils
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from utils.data_utils import SubtitleWrapper, normalize_string
import joblib as jl

import seaborn as sns
from sklearn.decomposition import PCA
from openTSNE import TSNE
import contextlib
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

debug = False

def init_model(args, lang_model, pose_dim, _device):
    n_frames = args.n_poses
    generator = Autoencoder_VQVAE(args, pose_dim, n_frames).to(_device)
    loss_fn = torch.nn.MSELoss()

    return generator, loss_fn




def train_epochs(args, train_data_loader, train_sim_dataset, test_data_loader, lang_model, pose_dim, trial_id=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss')]

    if args.use_derivitive=='True':
        pass

    # interval params
    print_interval = int(len(train_data_loader))
    save_model_epoch_interval = args.epochs

    # init model
    generator, loss_fn = init_model(args, lang_model, pose_dim, device)
    start_epoch = 1
    load_pretrained = False
    if load_pretrained:
        checkpoint_path_rnn = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, 20)
        args, generator, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
            checkpoint_path_rnn, device, 'autoencoder_vq')
        start_epoch = 20


    # Here, we freeze the encoder to train the decoder
    if args.autoencoder_freeze_encoder == 'True':
        rnn = utils.train_utils.load_checkpoint_and_model(
            args.autoencoder_checkpoint, device)
        args_rnn, rnn, loss_fn, lang_model, out_dim = rnn
        generator = rnn
        generator.freez_encoder()
        generator.autoencoder_fixed_weight = False
        for param in generator.decoder.parameters():
            param.requires_grad = True

    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, args)


    # To record loss and evaluations metric:
    val_metrics_list = []
    loss_list = []


    # training
    global_iter = 0

    train_res_recon_error = []
    train_res_perplexity = []
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    #     plot_embedding(args, generator, 0,
    #                    train_res_recon_error,
    #                    train_res_perplexity)
    for epoch in range(start_epoch, args.epochs+1):

        if args.autoencoder_vq == 'True':
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                plot_embedding(args, generator, epoch,
                               train_res_recon_error,
                               train_res_perplexity)

        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, args)
        val_metrics_list.append(val_metrics)
        # save model
        if epoch % save_model_epoch_interval == 0 and epoch > 0:
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'lang_model': lang_model,
                'pose_dim': pose_dim,
                'gen_dict': gen_state_dict,
                'val_metrics_list': val_metrics_list,
                'loss_list': loss_list

            }, save_name)

        # train iter
        iter_start_time = time.time()
        loss_epoch = AverageMeter('loss')
        loss_epoch.reset()
        # Todo: move to args
        train_labeled = False
        labeled_count = 20
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            encoded_input, encoded_output = data
            batch_size = encoded_output.size(0)

            encoded_input = encoded_input.to(device)
            encoded_output = encoded_output.to(device)

            # train

            if train_labeled:
                stack_pairs1, stack_pairs2, stack_labels = train_sim_dataset.get_labeled_(labeled_count)

                if debug:
                    print("STAAAAACK", stack_pairs1.shape)
                    print("STAAAAACK_LLL", stack_labels.shape)
                stack_pairs1 = stack_pairs1.to(device)
                stack_pairs2 = stack_pairs2.to(device)
                stack_labels = stack_labels.to(device)


                loss = train_iter_Autoencoder_ssl_seq2seq(args, epoch, encoded_input, encoded_output, generator,
                                                      gen_optimizer,
                                                          stack_pairs1, stack_pairs2, stack_labels)
            else:
                if args.autoencoder_vq == 'True':
                    loss, perplexity = train_iter_Autoencoder_VQ_seq2seq(args, epoch,
                                                          encoded_input, encoded_output, generator, gen_optimizer)
                    train_res_recon_error.append(loss['loss'])
                    train_res_perplexity.append(perplexity.item())
                else:
                    loss = train_iter_Autoencoder_VQ_seq2seq(args, epoch,
                                                                         encoded_input, encoded_output, generator,
                                                                         gen_optimizer)

            # Visualization for debug



            loss_epoch.update(loss['loss'], batch_size)
            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        # if args.autoencoder_vq == 'True':
                        #     print_summary += '\nperplexity,' + str(perplexity.detach().cpu().numpy())
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

        loss_list.append(loss_epoch.avg)

        # if args.autoencoder_vq == 'True':
        #     with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        #         plot_embedding(args, generator, epoch,
        #                        train_res_recon_error,
        #                        train_res_perplexity)

    plot_loss(args, all_eval_loss=val_metrics_list, all_train_loss=loss_list)


def evaluate_testset(test_data_loader, generator, loss_fn, args):
    # to evaluation mode
    generator.train(False)

    losses = AverageMeter('loss')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            in_poses, target_poses = data
            batch_size = in_poses.size(0)
            if debug:
                print("in_shape", in_poses.shape, "Target_poses", target_poses.shape)
            in_poses = in_poses.to(device)
            target_poses = target_poses.to(device)
            if args.autoencoder_vq == 'True':
                if args.autoencoder_vae== 'True':
                    out_poses, latent_, mue, logvar, loss_vq, perplexity_vq= generator(in_poses, target_poses)
                else:
                    out_poses, latent_, loss_vq, perplexity_vq = generator(in_poses, target_poses)
            else:
                if args.autoencoder_vae== 'True':
                    out_poses, latent_, mue, logvar = generator(in_poses, target_poses)
                else:
                    out_poses, latent_ = generator(in_poses, target_poses)
            # print("out_poses", out_poses.shape, "Target_poses", target_poses.shape)
            loss = loss_fn(out_poses, target_poses)
            losses.update(loss.item(), batch_size)

    # back to training mode
    generator.train(True)

    # print
    elapsed_time = time.time() - start
    logging.info('[VAL] loss: {:.3f} / {:.1f}s'.format(losses.avg, elapsed_time))

    return losses.avg



def plot_loss(args, all_eval_loss, all_train_loss):


    # X = np.arange(136-3)
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.bar(X + 0.00, all_train_loss[0], color='b', width=0.25)
    # ax.bar(X + 0.25, all_eval_loss[1], color='g', width=0.25)
    # plt.show()
    # plotting the second plot
    plt.plot(all_train_loss, label='Train loss')
    plt.plot(all_eval_loss, label='Evaluation loss')

    # Labeling the X-axis
    plt.xlabel('Epoch number')
    # Labeling the Y-axis
    plt.ylabel('Loss Average')
    # Give a title to the graph
    plt.title('Training/Evaluation Loss based on epoch number')

    # Show a legend on the plot
    plt.legend()

    plt.savefig(os.path.join(args.model_save_path, 'loss_plot.png'))
    plt.show()

def plot_embedding(args, rnn, epoch_num,
                   train_res_recon_error,
                   train_res_perplexity):
    os.makedirs(args.model_save_path + '/plots', exist_ok=True)

    w = rnn.vq_layer._embedding.weight.cpu().detach().numpy()
    # PCA
    try:
        pca = PCA(n_components=50)
        priciple_components = pca.fit(w)
        normalized_data = priciple_components.transform(w)
    except:
        normalized_data = w
    # TSNE

    MyTSNE = TSNE(
        n_components=2,
        perplexity=30,
        metric='euclidean',
        n_jobs=8,
        verbose=True
    )
    X_embedded = MyTSNE.fit(normalized_data)

    plt.figure(figsize=(16, 10))
    # palette = sns.color_palette("bright", k_component)
    # 2D

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1],
                    # hue=labels_,
                    legend=False)
    plt.title("Embedding visualization" + str(epoch_num))
    
    address = (args.model_save_path) + '/plots/Embedding_Epoch(' + str(epoch_num) +').png'
    plt.savefig(address)
    plt.clf()
    plt.show()


    # Reconstruction loss and perplexity
    print(train_res_perplexity)
    try:
        train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
        train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
    except:
        train_res_recon_error_smooth = train_res_recon_error  # savgol_filter(train_res_recon_error, 201, 7)
        train_res_perplexity_smooth = train_res_perplexity  # [0] #savgol_filter(train_res_perplexity, 201, 7)

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    val = "?"
    if len(train_res_recon_error_smooth)>0:
        val = str(train_res_recon_error_smooth[-1])
    ax.set_title('Smoothed NMSE: ' + val)
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(train_res_perplexity_smooth)

    val = "?"
    if len(train_res_perplexity_smooth)>0:
        val += str(train_res_perplexity_smooth[-1])
    ax.set_title('Smoothed Average codebook usage (perplexity): ' + val)
    ax.set_xlabel('iteration')

    address = (args.model_save_path) + '/plots/Error_Epoch(' + str(epoch_num) + ').png'

    plt.savefig(address)
    plt.clf()
    plt.show()




    pass

def save_sim_graph_gesture(args, pairs, data_original):
    save_result_path = args.model_save_path
    sim_count = 0
    dissim_count = 0
    for item in tqdm(pairs):
        if item[2] == 0:
            dissim_count += 1
            directory = str(save_result_path) + '/sim_graph/dis/' + str(dissim_count)
            os.makedirs(directory)
            make_bvh(directory, str(item[0]), data_original[item[0]])
            make_bvh(directory, str(item[1]), data_original[item[1]])
        else:
            sim_count = sim_count + 1
            directory = str(save_result_path) + '/sim_graph/sim/' + str(sim_count)
            os.makedirs(directory)
            make_bvh(directory, str(item[0]), data_original[item[0]])
            make_bvh(directory, str(item[1]), data_original[item[1]])
def make_bvh(save_path, filename_prefix, poses):
    writer = BVHWriter()
    pipeline = jl.load('../resource/data_pipe.sav')

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))

    for i in range(poses.shape[1]):
        out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal

    # rotation matrix to euler angles
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 9))
    out_poses = out_poses.reshape((out_poses.shape[0], out_poses.shape[1], 3, 3))
    out_euler = np.zeros((out_poses.shape[0], out_poses.shape[1] * 3))
    for i in range(out_poses.shape[0]):  # frames
        r = R.from_matrix(out_poses[i])
        out_euler[i] = r.as_euler('ZXY', degrees=True).flatten()

    bvh_data = pipeline.inverse_transform([out_euler])

    out_bvh_path = os.path.join(save_path, filename_prefix + '_generated.bvh')
    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)





def main(config):
    args = config['args']

    trial_id = None

    # random seed
    if args.random_seed >= 0:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info(pprint.pformat(vars(args)))

    # dataset
    # Todo: change it backt to the real train
    train_dataset = TrinityDataset_DAEed_Autoencoder(args, args.train_data_path[0],
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate,
                                   data_mean=args.data_mean, data_std=args.data_std)

    if args.use_similarity == 'True':
        print()
        train_dataset.creat_similarity_dataset(args.data_for_sim, args.similarity_labels)
    #     Sanity check
    #     save_sim_graph_gesture(args, train_dataset.pairwise_labels, train_dataset.data_original)
    #     print("Sanity check saved!")


    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              # collate_fn=word_seq_collate_fn
                              )


    val_dataset = TrinityDataset_DAEed_Autoencoder(args, args.val_data_path[0],
                                 n_poses=args.n_poses,
                                 subdivision_stride=args.subdivision_stride,
                                 pose_resampling_fps=args.motion_resampling_framerate,
                                 data_mean=args.data_mean, data_std=args.data_std)

    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             # collate_fn=word_seq_collate_fn
                             )

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    # lang_model = build_vocab('words', [train_dataset, val_dataset], vocab_cache_path, args.wordembed_path,
    #                          args.wordembed_dim)
    # train_dataset.set_lang_model(lang_model)
    # val_dataset.set_lang_model(lang_model)

    # train

    train_epochs(args, train_loader, train_dataset, test_loader, None,
                 pose_dim=args.rep_learning_dim, trial_id=trial_id)

    # Do inference

    checkpoint_path_rnn = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, args.epochs)
    checkpoint_path_DAE = args.rep_learning_checkpoint
    command = 'python inference_Autoencoder.py ' + checkpoint_path_DAE + ' ' + checkpoint_path_rnn
    os.system(command)

def save_config(_args):
    if not os.path.exists(_args.model_save_path):
        os.mkdir(_args.model_save_path)
    conf = open(_args.model_save_path + '/conf', 'w')
    str_conf = ''
    for arg in vars(_args):
        att = getattr(_args, arg)
        if isinstance(att, list):
            att = att[0]
        try:
            str_conf += str(arg) + '=' + str(att) + '\n'
        except:
            str_conf += str(arg) + '=' + 'None' + '\n'

    conf.write(str_conf)
    conf.flush()
    conf.close()

if __name__ == '__main__':
    # --config =../config/seq2seq.yml
    _args = parse_args()

    if _args.use_derivitive == 'True':
        _args.rep_learning_dim *= 2
    save_config(_args)
    main({'args': _args})
