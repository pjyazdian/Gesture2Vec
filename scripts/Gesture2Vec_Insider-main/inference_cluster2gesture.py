import argparse
import math
import pickle
import pprint
from pathlib import Path

import librosa
import numpy as np
import time

import torch
from scipy.signal import savgol_filter
import joblib as jl

import utils
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from tqdm import tqdm
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger

from data_loader.data_preprocessor import DataPreprocessor

from trinity_data_to_lmdb import process_bvh

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_gestures(args, DAE, rnn, bvh_file):
    poses, poses_mirror = process_bvh(bvh_file)

    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    out_poses = (poses_mirror - mean) / std

    target = torch.from_numpy(out_poses)
    # target = torch.unsqueeze(target,2)
    target = target.to(device).float()
    reconstructed = []
    # for i in range(len(out_poses)):
    #     input = torch.unsqueeze(target[i],0)
    #     current_out = pose_decoder(input)
    #     reconstructed.append(current_out)
    encoded = DAE.encoder(target)

    # Todo: move this to the input args
    use_derivitive = True
    if use_derivitive:
        diff = [(encoded[n, :] - encoded[n - 1, :]) for n in range(1, encoded.shape[0])]
        diff.insert(0, torch.zeros_like(encoded[0, :]))
        encoded = torch.hstack((encoded, torch.stack(diff)))
    # encoded = torch.squeeze(encoded, 2)
    # encoded = encoded.to('cpu')
    # encoded = encoded.detach().numpy()
    final_result = {}
    n_sample = 20
    for i in range(0, 40):
        input_seq = torch.from_numpy(np.repeat(i, n_sample)).to(device)
        input_pre_seq = encoded[i]
        output_seq = encoded[i:i + args.n_poses]
        output_seq = torch.cat([encoded[i:i + args.n_poses].unsqueeze(0) for rep in range(n_sample)])

        # run through decoder one time step at a time
        output_current = rnn(input_seq, output_seq)
        if use_derivitive:
            output_current = output_current[:, :, : output_current.shape[2] // 2]
        saved_shaped = output_current.shape
        output_current = torch.reshape(output_current, (saved_shaped[0]*saved_shaped[1], saved_shaped[2]))
        reconstructed_seq_DAE = DAE.decoder(output_current)
        reconstructed_seq_DAE = torch.reshape(reconstructed_seq_DAE, (saved_shaped[0], saved_shaped[1], -1))
        reconstructed_seq_DAE = reconstructed_seq_DAE.to('cpu')
        reconstructed_seq_DAE = reconstructed_seq_DAE.detach().numpy()
        final_result[i] = np.array(reconstructed_seq_DAE)


    return final_result

    '''
        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete the last part

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(out_seq)

    print('Avg. inference time: {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    out_poses = np.vstack(out_list)
    '''
    return out_poses


def main(checkpoint_path_DAE, checkpoint_path_rnn):
    args, DAE, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_DAE, device)
    args, rnn, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_rnn, device, 'c2g')
    pprint.pprint(vars(args))
    save_path = '../output/infer_sample'
    os.makedirs(save_path, exist_ok=True)

    # load lang_model
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    # prepare input
    # transcript = SubtitleWrapper(transcript_path).get()

    # inference
    bvh_file = '/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh'

    reconstructed = generate_gestures(args, DAE, rnn, bvh_file)

    # unnormalize
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    for i_cluster in range(0, 40):
        for j in range(0, 20):
            q = np.multiply(reconstructed[i_cluster][j], std) + mean
            reconstructed[i_cluster][j] = q


    # ........ infer

    # make a BVH
    for i_cluster in tqdm(range(0,40)):
        for j in range(0, 20):
            filename_prefix = "infered_" +  str(i_cluster) + "_" + str(j)
            make_bvh(save_path, filename_prefix, reconstructed[i_cluster][j])



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


def plot_loss(checkpoint_path_rnn):
    x = torch.load(checkpoint_path_rnn)
    all_eval_loss = x['val_metrics_list']
    all_train_loss = x['loss_list']
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

    # plt.savefig(os.path.join(args.model_save_path, 'loss_plot.png'))
    plt.show()
    # exit()


if __name__ == '__main__':
    '''

        ../output/DAE/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
        ../output/autoencoder/c2g/autoencode_fxw_zinput_checkpoint_100.bin

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path_DAE", type=Path)
    parser.add_argument("ckpt_path_Autoencode", type=Path)
    args = parser.parse_args()

    # plot_loss(args.ckpt_path_Autoencode)
    main(args.ckpt_path_DAE, args.ckpt_path_Autoencode)
