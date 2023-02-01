import argparse
import math
import os
import pickle
import pprint
from pathlib import Path

import librosa
import numpy as np
import time

import scipy
import torch
from scipy.signal import savgol_filter
import joblib as jl

import utils
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger

from data_loader.data_preprocessor import DataPreprocessor

from trinity_data_to_lmdb import process_bvh

import matplotlib.pyplot as plt


from trinity_data_to_lmdb import process_bvh as process_bvh_trinity
from twh_dataset_to_lmdb import process_bvh as process_bvh_twh
from twh_dataset_to_lmdb import process_bvh_rot_only as process_bvh_rot_only_twh
from twh_dataset_to_lmdb import process_bvh_rot_only_Taras as process_bvh_rot_only_Taras
from twh_dataset_to_lmdb import process_bvh_test1 as process_bvh_rot_test1
from inference_DAE import feat2bvh, make_bvh_Trinity

from basis_expansions import (Binner,
                              GaussianKernel,
                              Polynomial,
                              LinearSpline,
                              CubicSpline,
                              NaturalCubicSpline)
from sklearn.linear_model import LinearRegression
import csaps
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_TYPE = 'TWH'

def generate_gestures(args, DAE, rnn, bvh_file):

    if DATASET_TYPE == 'Trinity':
        poses, poses_mirror = process_bvh_trinity(bvh_file)
    elif DATASET_TYPE == 'TWH':
        poses = process_bvh_rot_test1(bvh_file)
        poses_mirror = poses[250:500]

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
    use_derivitive = False
    if use_derivitive:
        diff = [(encoded[n, :] - encoded[n - 1, :]) for n in range(1, encoded.shape[0])]
        diff.insert(0, torch.zeros_like(encoded[0, :]))
        encoded = torch.hstack((encoded, torch.stack(diff)))
    # encoded = torch.squeeze(encoded, 2)
    # encoded = encoded.to('cpu')
    # encoded = encoded.detach().numpy()
    # Todo: remove this.

    all_frames_from_rnn = None
    decoder_input = None
    for i in range(0, len(encoded)-args.n_poses, args.n_poses):
        input_seq = encoded[i:i+args.n_poses]
        input_pre_seq = encoded[i]
        output_seq = encoded[i:i+args.n_poses]

        input_seq = torch.unsqueeze(input_seq, 0)
        input_seq = input_seq.transpose(0, 1)

        output_seq = torch.unsqueeze(output_seq, 0)
        output_seq = output_seq.transpose(0, 1)

        reconstructed_rnn = torch.zeros(args.n_poses, output_seq.size(1), rnn.decoder.output_size).to(output_seq.device)

        # Todo: For CNN case
        # reconstructed_rnn = torch.zeros(args.n_poses, output_seq.size(1), 135).to(output_seq.device)

        # run words through encoder
        encoder_outputs, encoder_hidden = rnn.encoder(input_seq, None)
        decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder

        if rnn.VAE:
            decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder
            # [2, 128, 200]
            # print("decoder_hidden!!! org", decoder_hidden.shape)
            decoder_hidden = decoder_hidden.transpose(1, 0).contiguous() # [128, 2, 200]
            decoder_hidden = torch.reshape(decoder_hidden, (decoder_hidden.shape[0], -1))
            mean = rnn.VAE_fc_mean(decoder_hidden)
            logvar = rnn.VAE_fc_std(decoder_hidden)
            z = rnn.reparameterize(mean, logvar, train=False)
            z = rnn.VAE_fc_decoder(z)
            decoder_hidden = z.reshape(decoder_hidden.shape[0],
                                       rnn.decoder.n_layers, -1)
            decoder_hidden = decoder_hidden.transpose(1 ,0).contiguous()
            # print("decoder_hidden!!! modified", decoder_hidden.shape)
            decoder_first_hidden = decoder_hidden
        else:
            decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder
            min_ = torch.min(encoder_hidden)
            max_ = torch.max(encoder_hidden)
            # print("decoder_hidden!!! not VAE ", decoder_hidden.shape)

        if args.autoencoder_vq == 'True':
            loss_vq, quantized, perplexity_vq, encodings = rnn.vq_layer(decoder_hidden)
            ifx =  torch.eq(quantized,decoder_hidden)

            decoder_hidden = quantized

        if rnn.CNN:
            print(decoder_hidden.shape)

            outputs, hidden = rnn.decoder(decoder_hidden)
            shape = outputs.shape
            # print(shape)
            outputs = outputs.transpose(1,2)
            # print(outputs.shape)
            shape = outputs.shape
            outputs = outputs.reshape(shape[0]*shape[1], shape[2])
            # print(outputs.shape)
            outputs = rnn.out_layer_decoder(outputs)
            outputs = outputs.reshape(shape[0], shape[1], -1)
            reconstructed_rnn = outputs.transpose(0, 1)
            # print(outputs.shape)
        else:
            # run through decoder one time step at a time
            if decoder_input is None:
                decoder_input = output_seq[0]  # initial pose from the dataset
            reconstructed_rnn[0] = decoder_input

            # repeat = 0
            for rep in range(5):
                decoder_output, decoder_hidden, _ = rnn.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                                None)


            for t in range(1, rnn.n_frames):
                if not rnn.autoencoder_conditioned:
                    decoder_input = torch.zeros_like(decoder_input)
                decoder_output, decoder_hidden, _ = rnn.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                                 None)
                reconstructed_rnn[t] = decoder_output

                if t < rnn.n_pre_poses:
                    decoder_input = output_seq[t]  # next input is current target
                else:
                    decoder_input = decoder_output  # next input is current prediction
                if not rnn.autoencoder_conditioned:
                    decoder_input = torch.zeros_like(decoder_input)




        if all_frames_from_rnn == None:
            all_frames_from_rnn = reconstructed_rnn.transpose(0,1)
        else:
            all_frames_from_rnn = torch.cat((all_frames_from_rnn, reconstructed_rnn.transpose(0, 1)), 1)


    #     Todo: decode DAE
    all_frames_from_rnn = torch.squeeze(all_frames_from_rnn, 0)
    if use_derivitive:
        all_frames_from_rnn = all_frames_from_rnn[:, 0: all_frames_from_rnn.shape[1]//2]
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

Flag_VQ = True


def plot_embedding(rnn):
    import seaborn as sns
    from sklearn.decomposition import PCA
    from openTSNE import TSNE


    try:

        # Plot Pre linear kernel
        w = rnn.vq_layer.pre_linear.weight.cpu().detach().numpy()
        plt.imshow(w, cmap='Reds')
        plt.show()


        w = rnn.vq_layer._embedding.weight.cpu().detach().numpy()
        # PCA

        # plot embedding as image
        plt.imshow(w)
        plt.title("Embeddings as image")
        plt.show()

        pca = PCA(n_components=50)
        priciple_components = pca.fit(w)
        normalized_data = priciple_components.transform(w)
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
        plt.title("Embedding visualization")
        address = os.path.dirname(args.ckpt_path_Autoencode) + '/Embedding.png'
        plt.savefig(address)
        plt.show()
        pass
    except:
        pass

def check_prototypes(model):
    Embeddings = model.vq_layer._embedding
    weights = Embeddings.weight
    print(weights.shape)
    dists = torch.cdist(weights, weights)
    dists = dists.cpu().detach().numpy()
    plt.imshow(dists)
    plt.show()

def main(checkpoint_path_DAE, checkpoint_path_rnn):

    # Frame level autoencoder
    args, DAE, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_DAE, device, 'DAE')

    # Sequence level autoencoder
    if not Flag_VQ:
        args, rnn, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
            checkpoint_path_rnn, device, 'autoencoder')
    else:
        args, rnn, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
            checkpoint_path_rnn, device, 'autoencoder_vq')
        plot_embedding(rnn)
        if args.autoencoder_vq == 'True':
            check_prototypes(rnn)

    pprint.pprint(vars(args))
    save_path = '../output/infer_sample'
    save_path = os.path.dirname(checkpoint_path_rnn) + '/infer_sample'
    os.makedirs(save_path, exist_ok=True)

    # load lang_model
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    # prepare input
    # transcript = SubtitleWrapper(transcript_path).get()

    # inference
    for infer_i in range(1, 2):
        if DATASET_TYPE == 'Trinity':
            bvh_file = '/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq0'
            org_poses, reconstructed = generate_gestures(args, DAE, rnn, bvh_file + str(infer_i).zfill(2) + '.bvh')
        elif DATASET_TYPE == 'TWH':
            pre_bvh_pth = "/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/GENEA/Co-Speech_Gesture_Generation/dataset/dataset_v1/val/bvh/"
            bvh_file_names_list = ['val_2022_v1_038.bvh', 'val_2022_v1_012.bvh']
            org_poses, reconstructed = generate_gestures(args, DAE, rnn,pre_bvh_pth + bvh_file_names_list[infer_i])

        for iq in range(5):
            plt.plot(reconstructed[:, iq])
        plt.show()
        plt.clf()
        # smoothing motion transition
        for offset in range(30, len(reconstructed), 30):
            for index in range(10):
                n = 10
                prev = reconstructed[offset - index ]
                next = reconstructed[offset + index]
                reconstructed[offset + index] = prev * (n - index) / (n + 1) + next * (index + 1) / (n + 1)

        # upsampling




        # unnormalize
        mean = np.array(args.data_mean).squeeze()
        std = np.array(args.data_std).squeeze()
        std = np.clip(std, a_min=0.01, a_max=None)
        reconstructed = np.multiply(reconstructed, std) + mean
        org_poses = np.multiply(org_poses, std) + mean

        # ........ infer


        # for i in range(0, len(reconstructed), 30):
        #     make_bvh(save_path, 'test_' + str(i//30)+ "_f_" + str(i), reconstructed[i:i+30, :])

        # make a BVH
        filename_prefix = '{}'.format("TestSeq{}".format(infer_i).zfill(3))
        make_bvh(save_path, filename_prefix, org_poses)
        filename_prefix = '{}'.format("test_reconstructed_" + str(infer_i))
        make_bvh(save_path, filename_prefix, reconstructed)


def smoothing_function(method, poses):

    if method == 'moving_average':
        q = np.zeros_like(poses)
        window_len = 10
        for i in range(poses.shape[1]):
            for j in range(poses.shape[0]):
                neighbours = []
                for w in range(-window_len, window_len):
                    if j + w >= 0 and j + w < poses.shape[0]:
                        neighbours.append(poses[j + w, i])

                q[j, i] = np.array(neighbours).mean()
        poses = q
    if method == 'convolution':
        window_len = 10

        for i in range(poses.shape[1]):
            a = poses[window_len-1:0:-1, i]
            b = poses[:, i]
            c = poses[-1:-window_len:-1, i]
            s = np.r_[a,b,c]

            w = np.ones(window_len, 'd')
            y = np.convolve(w / w.sum(), s, mode='valid')
            zz = y[0:-window_len+1]
            poses[:, i] = zz
    if method == 'upsample':
        q = np.zeros((poses.shape[0] * 2, poses.shape[1]))
        q = np.zeros((poses.shape[0] * 2, poses.shape[1]))
        for i in range(poses.shape[1]):
            q[:, i] = scipy.signal.resample(poses[:, i], poses.shape[0] * 2, domain='time')
        poses = q
    if method == 'cubic':
        win = 10
        def make_gaussian_regression(n_centers):
            return Pipeline([
                ('binner', GaussianKernel(0, 1, n_centers=n_centers, bandwidth=0.1)),
                ('regression', LinearRegression(fit_intercept=True))
            ])

        def make_polynomial_regression(degree):
            return Pipeline([
                ('std', StandardScaler()),
                ('poly', Polynomial(degree=degree)),
                ('regression', LinearRegression(fit_intercept=True))
            ])
        x = np.array(range(0, win))
        for offset in range(0, poses.shape[0]-win, win):
            for joint in range(poses.shape[1]):
                gr = make_gaussian_regression(1)
                gr.fit(x, poses[offset:offset+win, joint])
                plt.scatter(x, poses[offset:offset+win, joint], color='red')
                plt.scatter(x, gr.predict(x), color='blue')
                plt.plot(poses[offset:offset + win, joint], color='red')
                plt.plot(gr.predict(x), color='blue')
                poses[offset:offset+win, joint] = gr.predict(x)
                # plt.show()
    if method == 'spline':

        # deriv = []
        # for i in range(2, poses.shape[0]):
        #     deriv.append(np.sum(np.abs(poses[i] - poses[i-1])))
        # plt.plot(deriv)
        # plt.title("Derivitive")
        # # plt.show()

        win =poses.shape[0]
        upsample_in = False
        smooth_f = 0.5
        x = np.array(range(0, win))
        x_up = np.linspace(0, win, win*2, endpoint=True)
        if upsample_in:
            q = np.zeros((poses.shape[0] * 2, poses.shape[1]))
        else:
            q = np.zeros((poses.shape[0], poses.shape[1]))
        test = range(0, poses.shape[0] - win, win//4)
        for offset in tqdm(range(0, poses.shape[0], win)):
            if offset == 0:
                print("if offset == 330:")
            for joint in range(poses.shape[1]):
                # gr = csaps.UnivariateCubicSmoothingSpline(x, poses[offset:offset + win, joint],
                #                                           smooth=0.85)
                # predicted = csaps.csaps(x, poses[offset:offset + win, joint], x, smooth=-3)
                if upsample_in:
                    predicted = csaps.csaps(x, poses[:, joint], x_up, smooth=smooth_f)
                    predicted = csaps.csaps(x, predicted, x_up, smooth=smooth_f)

                else:
                    predicted = csaps.csaps(x, poses[:, joint], x, smooth=smooth_f)

                q[offset:offset + win * 2, joint] = (predicted)

                # if offset==0 and joint==0:
                #     # print()
                #     # plt.scatter(x, poses[offset:offset + win, joint], color='red')
                #     # plt.scatter(x, predicted, color='blue')
                #     plt.plot(poses[offset:offset + win, joint], color='red')
                #     plt.plot(predicted, color='blue')
                #     plt.title("smooth="+str(smooth_f))
                #     # plt.show()
            # exit()
            poses = q

    # deriv = []
    # for i in range(2, poses.shape[0]):
    #     deriv.append(np.sum(np.abs(poses[i] - poses[i - 1])))
    # plt.plot(deriv)
    # plt.title("Derivitive_after")
    # # plt.show()

    return poses





def make_bvh(save_path, filename_prefix, poses):

    if DATASET_TYPE=='TWH':
        return make_bvh_Trinity(save_path, filename_prefix, poses)
        return feat2bvh(save_path, filename_prefix, poses)

    writer = BVHWriter()
    pipeline = jl.load('../resource/data_pipe.sav')

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))


    # out_poses = np.zeros_like(poses)
    for i in range(poses.shape[1]):
        out_poses[:, i] = savgol_filter(poses[:, i], 15, polyorder=3)  # NOTE: smoothing on rotation matrices is not optimal



    # rotation matrix to euler angles
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 9))
    out_poses = out_poses.reshape((out_poses.shape[0], out_poses.shape[1], 3, 3))
    out_euler = np.zeros((out_poses.shape[0], out_poses.shape[1] * 3))
    for i in range(out_poses.shape[0]):  # frames
        r = R.from_matrix(out_poses[i])
        out_euler[i] = r.as_euler('ZXY', degrees=True).flatten()

    out_euler = smoothing_function('spline',out_euler)

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

    address = os.path.dirname(args.ckpt_path_Autoencode) + '/loss_plot.png'
    plt.savefig(address)
    # plt.savefig(os.path.join(args.model_save_path, 'loss_plot.png'))
    plt.show()
    # exit()

if __name__ == '__main__':
    '''
    
        ../output/DAE/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
        ../output/autoencoder/without_att_fxw_zinput/autoencode_fxw_zinput_checkpoint_100.bin
        
        
        ../output/DAE/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
        ../output/autoencoder/VAE/Dautoencode_fxw_zinput_checkpoint_100.bin
        
        
        ../output/DAE/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
../output/autoencoder/VAE+sim/autoencoder_decoder_trained_cont/autoencoder_decoder_trained_cont_checkpoint_040.bin
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path_DAE", type=Path)
    parser.add_argument("ckpt_path_Autoencode", type=Path)
    args = parser.parse_args()

    plot_loss(args.ckpt_path_Autoencode)

    main(args.ckpt_path_DAE, args.ckpt_path_Autoencode)
