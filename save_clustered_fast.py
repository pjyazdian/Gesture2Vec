import argparse
import glob
import math
import os
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
    # encoded = torch.squeeze(encoded, 2)
    # encoded = encoded.to('cpu')
    # encoded = encoded.detach().numpy()
    all_frames_from_rnn = None
    for i in range(0, len(encoded), args.n_poses):
        input_seq = encoded[i:i+args.n_poses]
        input_pre_seq = encoded[i]
        output_seq = encoded[i:i+args.n_poses]

        input_seq = torch.unsqueeze(input_seq, 0)
        input_seq = input_seq.transpose(0, 1)

        output_seq = torch.unsqueeze(output_seq, 0)
        output_seq = output_seq.transpose(0, 1)
        reconstructed_rnn = torch.zeros(args.n_poses, output_seq.size(1), rnn.decoder.output_size).to(output_seq.device)
        # run words through encoder
        encoder_outputs, encoder_hidden = rnn.encoder(input_seq, None)
        decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder

        # run through decoder one time step at a time
        decoder_input = output_seq[0]  # initial pose from the dataset
        reconstructed_rnn[0] = decoder_input

        for t in range(1, rnn.n_frames):
            decoder_output, decoder_hidden, _ = rnn.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                             None)
            reconstructed_rnn[t] = decoder_output

            if t < rnn.n_pre_poses:
                decoder_input = output_seq[t]  # next input is current target
            else:
                decoder_input = decoder_output  # next input is current prediction
        if all_frames_from_rnn == None:
            all_frames_from_rnn = reconstructed_rnn.transpose(0,1)
        else:
            all_frames_from_rnn = torch.cat((all_frames_from_rnn, reconstructed_rnn.transpose(0, 1)), 1)


    #     Todo: decode DAE
    all_frames_from_rnn = torch.squeeze(all_frames_from_rnn, 0)
    reconstructed_seq_DAE = DAE.decoder(all_frames_from_rnn)
    # reconstructed_seq_DAE = torch.squeeze(reconstructed_seq_DAE, 2)
    reconstructed_seq_DAE = reconstructed_seq_DAE.to('cpu')
    reconstructed_seq_DAE = reconstructed_seq_DAE.detach().numpy()


    return out_poses, np.array(reconstructed_seq_DAE)

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
        checkpoint_path_rnn, device)
    pprint.pprint(vars(args))
    save_path = '../output/infer_sample'
    os.makedirs(save_path, exist_ok=True)

    # load lang_model
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    # prepare input
    # transcript = SubtitleWrapper(transcript_path).get()

    k = 0
    kmeans_labels = pickle.load(open('../output/kmeans_labels.bin', 'rb'))
    # inference

    gesture_path = '/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Training_data/Motion'

    bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
    for bvh_file in tqdm(bvh_files):
        name = os.path.split(bvh_file)[1][:-4]
        print("Processing", name)
        org_poses, reconstructed = generate_gestures(args, DAE, rnn, bvh_file)

        # unnormalize
        mean = np.array(args.data_mean).squeeze()
        std = np.array(args.data_std).squeeze()
        std = np.clip(std, a_min=0.01, a_max=None)
        reconstructed = np.multiply(reconstructed, std) + mean
        org_poses = np.multiply(org_poses, std) + mean

        for j in range(0, len(reconstructed), args.n_poses):
            save_path = '../output/clusters/' + str(kmeans_labels[k])
            k = k+1
            # make a BVH
            filename_prefix = '{}'.format("test_original")
            make_bvh(save_path, filename_prefix, org_poses[j:j+args.n_poses])
            filename_prefix = '{}'.format("test_reconstructed")
            make_bvh(save_path, filename_prefix, reconstructed[j:j+args.n_poses])


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path_DAE", type=Path)
    parser.add_argument("ckpt_path_Autoencode", type=Path)
    args = parser.parse_args()
    labels = pickle.load(open('../output/kmeans_labels.bin', 'rb'))
    main(args.ckpt_path_DAE, args.ckpt_path_Autoencode)
