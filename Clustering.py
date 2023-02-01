import argparse
import glob
import math
import pickle
import pprint
from pathlib import Path
from collections import Counter

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import string_
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

import torch
from scipy.signal import savgol_filter
import joblib as jl

import utils
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger

from data_loader.data_preprocessor import DataPreprocessor

from trinity_data_to_lmdb import process_bvh as process_bvh_trinity
from twh_dataset_to_lmdb import process_bvh_rot_only_Taras as process_bvh_rot_only_Taras

from twh_dataset_to_lmdb import process_bvh_test1 as process_bvh_rot_test1
from inference_DAE import make_bvh_Trinity


from inference_DAE import feat2bvh
import os.path
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
import random
from scipy import stats


from sklearn.decomposition import PCA
from openTSNE import TSNE
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_TYPE = 'TWH'

def generate_gestures_latent_dataset(args, DAE, rnn, bvh_file):

    if DATA_TYPE=='Trinity':
        poses, poses_mirror = process_bvh(bvh_file)
    elif DATA_TYPE == 'TWH':
        poses = process_bvh_rot_test1(bvh_file)
        poses_mirror = poses
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    out_poses = (np.copy(poses_mirror) - mean) / std

    target = torch.from_numpy(out_poses)
    # target = torch.unsqueeze(target,2)
    target = target.to(device).float()
    reconstructed = []
    # for i in range(len(out_poses)):
    #     input = torch.unsqueeze(target[i],0)
    #     current_out = pose_decoder(input)
    #     reconstructed.append(current_out)
    if DAE.encoder == None:
        encoded = target
    else:
        encoded = DAE.encoder(target)
    # encoded = torch.squeeze(encoded, 2)
    # encoded = encoded.to('cpu')
    # encoded = encoded.detach().numpy()
    all_frames_from_rnn = None
    all_sequences_poses_latent = []
    for i in range(0, len(encoded), args.subdivision_stride):
        current_dict = dict()
        input_seq = encoded[i:i+args.n_poses]
        if len(input_seq)<args.n_poses:
            continue
        # Todo: I did sanity check and 'original' should be normalized
        current_dict['original'] = poses_mirror[i: i+args.n_poses]
        current_dict['latent_linear'] = encoded[i: i + args.n_poses].detach().clone().to('cpu').numpy()




        input_pre_seq = encoded[i]
        output_seq = encoded[i:i+args.n_poses]

        input_seq = torch.unsqueeze(input_seq, 0)
        input_seq = input_seq.transpose(0, 1)
        # Todo: use args
        # use_drivitive=True
        if args.use_derivitive == 'True':
            diff = [(input_seq[n, :] - input_seq[n - 1, :]) for n in range(1, input_seq.shape[0])]
            diff.insert(0, torch.zeros_like(input_seq[0, :]))
            input_seq = torch.cat((input_seq, torch.stack(diff)), dim=2)


        output_seq = torch.unsqueeze(output_seq, 0)
        output_seq = output_seq.transpose(0, 1)
        # reconstructed_rnn = torch.zeros(args.n_poses, output_seq.size(1), rnn.decoder.output_size)\
        #     .to(output_seq.device)
        reconstructed_rnn = torch.zeros(args.n_poses, output_seq.size(1), 135) \
            .to(output_seq.device)
        # run words through encoder
        encoder_outputs, encoder_hidden = rnn.encoder(input_seq, None)

        if rnn.VAE:
            decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder
            # [2, 128, 200]
            # print("decoder_hidden!!! org", decoder_hidden.shape)
            decoder_hidden = decoder_hidden.transpose(1, 0).contiguous()  # [128, 2, 200]
            decoder_hidden = torch.reshape(decoder_hidden, (decoder_hidden.shape[0], -1))
            mean = rnn.VAE_fc_mean(decoder_hidden)
            logvar = rnn.VAE_fc_std(decoder_hidden)
            z = rnn.reparameterize(mean, logvar, train=False)
            z = rnn.VAE_fc_decoder(z)
            decoder_hidden = z.reshape(decoder_hidden.shape[0],
                                       rnn.decoder.n_layers, -1)
            decoder_hidden = decoder_hidden.transpose(1, 0).contiguous()
        else:
            decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder
        try:
            if rnn.vq:
                loss_vq, quantized, perplexity_vq, encodings = rnn.vq_layer(decoder_hidden)
                encodings = encodings.detach().cpu().numpy()
                quantized_indices = np.argmax(encodings, axis=1)
                current_dict['quantized_indices'] = quantized_indices


        except:
            # print("Skip VQ: Trained using the old autoencoder without VQ option!")
            pass
        current_dict['latent_rnn'] = torch.squeeze(decoder_hidden.detach().clone(), 1).to('cpu').numpy()

        all_sequences_poses_latent.append(current_dict)
        # run through decoder one time step at a time
    #     decoder_input = output_seq[0]  # initial pose from the dataset
    #     reconstructed_rnn[0] = decoder_input
    #
    #     for t in range(1, rnn.n_frames):
    #         decoder_output, decoder_hidden, _ = rnn.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
    #                                                          None)
    #         reconstructed_rnn[t] = decoder_output
    #
    #         if t < rnn.n_pre_poses:
    #             decoder_input = output_seq[t]  # next input is current target
    #         else:
    #             decoder_input = decoder_output  # next input is current prediction
    #     if all_frames_from_rnn == None:
    #         all_frames_from_rnn = reconstructed_rnn.transpose(0,1)
    #     else:
    #         all_frames_from_rnn = torch.cat((all_frames_from_rnn, reconstructed_rnn.transpose(0, 1)), 1)
    #
    #
    # #     Todo: decode DAE
    # all_frames_from_rnn = torch.squeeze(all_frames_from_rnn, 0)
    # reconstructed_seq_DAE = DAE.decoder(all_frames_from_rnn)
    # # reconstructed_seq_DAE = torch.squeeze(reconstructed_seq_DAE, 2)
    # reconstructed_seq_DAE = reconstructed_seq_DAE.to('cpu')
    # reconstructed_seq_DAE = reconstructed_seq_DAE.detach().numpy()



    return all_sequences_poses_latent

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




def make_VQ_Centers(checkpoint_path_DAE, checkpoint_path_rnn, save_path):
    args, DAE, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_DAE, device, 'DAE')
    args, rnn, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_rnn, device, 'autoencoder_vq')

    for index, quantized in enumerate(rnn.vq_layer._embedding.weight):

        # quantized = torch.reshape(quantized, [2, 1, 200]).contiguous()
        quantized = torch.reshape(quantized, [args.n_layers, 1, args.hidden_size]).contiguous()
        decoder_hidden = quantized.clone()

        # run through decoder one time step at a time
        decoder_input = torch.zeros(1, rnn.decoder.output_size).to(decoder_hidden.device)# initial pose from the dataset
        reconstructed_rnn = torch.zeros(args.n_poses, 1, rnn.decoder.output_size).to(decoder_hidden.device)

        reconstructed_rnn[0] = decoder_input

        # repeat = 0
        for rep in range(5):
            decoder_output, decoder_hidden, _ = rnn.decoder(None, decoder_input, decoder_hidden, quantized,
                                                            None)
        all_frames_from_rnn = None
        for t in range(1, rnn.n_frames):
            if not rnn.autoencoder_conditioned:
                decoder_input = torch.zeros_like(decoder_input)
            decoder_output, decoder_hidden, _ = rnn.decoder(None, decoder_input, decoder_hidden, quantized,
                                                            None)
            reconstructed_rnn[t] = decoder_output

            if t < rnn.n_pre_poses:
                decoder_input = output_seq[t]  # next input is current target
            else:
                decoder_input = decoder_output  # next input is current prediction
            if not rnn.autoencoder_conditioned:
                decoder_input = torch.zeros_like(decoder_input)

        if all_frames_from_rnn == None:
            all_frames_from_rnn = reconstructed_rnn.transpose(0, 1)
        else:
            all_frames_from_rnn = torch.cat((all_frames_from_rnn, reconstructed_rnn.transpose(0, 1)), 1)

        #     Todo: decode DAE

        use_derivitive = False
        all_frames_from_rnn = torch.squeeze(all_frames_from_rnn, 0)
        if args.use_derivitive == True:
            all_frames_from_rnn = all_frames_from_rnn[:, 0: all_frames_from_rnn.shape[1] // 2]
        reconstructed_seq_DAE = DAE.decoder(all_frames_from_rnn)
        # reconstructed_seq_DAE = torch.squeeze(reconstructed_seq_DAE, 2)
        reconstructed_seq_DAE = reconstructed_seq_DAE.to('cpu')
        reconstructed_seq_DAE = reconstructed_seq_DAE.detach().numpy()

        # Todo: save the reconstructed
        # unnormalize
        mean = np.array(args.data_mean).squeeze()
        std = np.array(args.data_std).squeeze()
        std = np.clip(std, a_min=0.01, a_max=None)
        reconstructed_seq_DAE = np.multiply(reconstructed_seq_DAE, std) + mean


        os.makedirs(save_path + '/Centers/', exist_ok=True)
        filename_prefix = 'Center_{}'.format(index)
        if DATA_TYPE=='Trinity':
            make_bvh(save_path + '/Centers/', filename_prefix, reconstructed_seq_DAE)
        elif DATA_TYPE=='TWH':
            make_bvh_Trinity(save_path + '/Centers/', filename_prefix, reconstructed_seq_DAE)

def maake_dataset(checkpoint_path_DAE, checkpoint_path_rnn, pickle_adress):
    args, DAE, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_DAE, device, 'DAE')
    args, rnn, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_rnn, device, 'autoencoder_vq')
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
    all_seq_real_latent = []
    gesture_path = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/GENEA/Co-Speech_Gesture_Generation/dataset/dataset_v1/trn/bvh'

    bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))

    count = 0

    for bvh_file in tqdm(bvh_files):
        count += 1
        if count > 20:
            print("*** More than 100 files")
            break
        name = os.path.split(bvh_file)[1][:-4]
        print("Processing", name)

        # bvh_file = '/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh'

        returned_org_latent = generate_gestures_latent_dataset(args, DAE, rnn, bvh_file)
        all_seq_real_latent.append(returned_org_latent)
        # break
    pickle.dump(all_seq_real_latent, open(pickle_adress, 'wb'))

    return
    '''
    # unnormalize
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    reconstructed = np.multiply(reconstructed, std) + mean
    org_poses = np.multiply(org_poses, std) + mean

    # ........ infer



    # make a BVH
    filename_prefix = '{}'.format("test_original")
    make_bvh(save_path, filename_prefix, org_poses)
    filename_prefix = '{}'.format("test_reconstructed")
    make_bvh(save_path, filename_prefix, reconstructed)
    '''

def maake_dataset_for_inference(checkpoint_path_DAE, checkpoint_path_rnn, bvh_file):
    args, DAE, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_DAE, device, 'DAE')
    args, rnn, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path_rnn, device, 'autoencoder_vq')
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
    all_seq_real_latent = []
    gesture_path = '/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Training_data/Motion'

    # bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))

    name = os.path.split(bvh_file)[1][:-4]
    print("Processing", name)

    # bvh_file = '/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh'

    if args.autoencoder_vq == 'True':
        returned_org_latent = generate_gestures_latent_dataset(args, DAE, rnn, bvh_file)
    else:
        returned_org_latent = generate_gestures_latent_dataset(args, DAE, rnn, bvh_file)
    return returned_org_latent


def make_bvh(save_path, filename_prefix, poses):
    writer = BVHWriter()
    pipeline = jl.load('../resource/data_pipe.sav')

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))

    try:
        for i in range(poses.shape[1]):
            out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal
    except:
        pass
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

Flag_VQ = True


def calculate_distances(data_latent_rnn_ndarray):
    print("Calculatiing neighbour feature distance...")
    # print("Skipping neighbour feature distance calculation...")
    # return
    # calculate representation quality
    # Implementation of representation space metric
    str_rep = '=====Representation similarity at the embedding space.=====\n\n'
    all_dist = pdist(data_latent_rnn_ndarray)
    avg_dist_total = np.mean(all_dist)
    for method in ['all', 'random']:
        print("Latent calculation method: ", method)
        if method == 'all':
            indicies = range(2, len(data_latent_rnn_ndarray) - 2)
        else:
            indicies = random.sample(range(2, len(data_latent_rnn_ndarray) - 2), 1000)

        neighbour_20fs = []
        neighbour_10fs = []
        # for i in range(2, len(data_latent_rnn_ndarray) - 2):
        for i in tqdm(indicies):
            n20s = (np.sqrt(np.sum((data_latent_rnn_ndarray[i - 1]
                                  - data_latent_rnn_ndarray[i]) ** 2, axis=0)) \
                  + np.sqrt(np.sum((data_latent_rnn_ndarray[i + 1]
                                    - data_latent_rnn_ndarray[i]) ** 2, axis=0))
                  ) / 2
            neighbour_20fs.append(n20s)

            n10s = (np.sqrt(np.sum((data_latent_rnn_ndarray[i - 2]
                                  - data_latent_rnn_ndarray[i]) ** 2, axis=0)) \
                  + np.sqrt(np.sum((data_latent_rnn_ndarray[i + 2]
                                    - data_latent_rnn_ndarray[i]) ** 2, axis=0))
                  ) / 2
            neighbour_10fs.append(n10s)

        avg_20fs = np.mean(neighbour_20fs)
        std_20fs = np.std(neighbour_20fs)

        avg_10fs = np.mean(neighbour_10fs)
        std_10fs = np.std(neighbour_10fs)




        neighbour_20fs /= avg_dist_total
        neighbour_10fs /= avg_dist_total
        normal_avg_20fs = np.mean(neighbour_20fs)
        normal_std_20fs = np.std(neighbour_20fs)
        normal_avg_10fs = np.mean(neighbour_10fs)
        normal_std_10fs = np.std(neighbour_10fs)

        str_rep += '\n\n==========================\nMethod: {}\n'.format(method)
        str_rep += "Representation Similarity metric\n" + \
                  "avg_20fs={} -- std_20fs={}\navg_10fs={} -- std_10fs={}\n" \
                      .format(avg_20fs, std_20fs, avg_10fs, std_10fs)
        str_rep += "\nStandardized:\n" + "average distance total: " \
                   + str(avg_dist_total) \
                   + "\nstandardized distances\n" \
                   + "normal_avg_20fs={} -- normal_std_20fs={}\nnormal_avg_10fs={} -- normal_std_10fs={}\n" \
                       .format(normal_avg_20fs, normal_std_20fs, normal_avg_10fs, normal_std_10fs)
        str_rep += '\n\n'
    print(str_rep)
    metric_file = open(save_result_path + "/Rep_distance.txt", 'w')
    metric_file.write(str_rep)
    metric_file.flush()
    metric_file.close()



def cluster(loaded, eps, min_samples, save_result_path):

    data_latent_rnn = []
    data_latent_linear = np.zeros([len(loaded), loaded[0]['latent_linear'].shape[0],
                                   loaded[0]['latent_linear'].shape[1]])
    data_latent_linear_listwise = []
    data_original = []
    data_quantized_indicies = []
    count = len(loaded)
    # count = 4000
    for i in range(count):


        # 1

        current_latent_linear = loaded[i]['latent_linear']
        current_latent_rnn = loaded[i]['latent_rnn']
        current_original = loaded[i]['original']
        if Flag_VQ:
            current_quantized_indicies = loaded[i]['quantized_indices'][0]

        if len(current_original) !=len(loaded[0]['original']):
            continue


        # 2
        # current_latent_linear = np.hstack(current_latent_linear)
        current_latent_rnn = np.hstack(current_latent_rnn)
        # current_original = np.hstack(current_original)

        # 3
        data_latent_linear[i] = (current_latent_linear)
        data_latent_linear_listwise.append(current_latent_linear)
        data_latent_rnn.append(current_latent_rnn)
        data_original.append(current_original)
        if Flag_VQ:
            data_quantized_indicies.append(current_quantized_indicies)
    # Should be constructed here since it is not obvious how many we will have at the end.
    data_latent_linear = np.zeros([len(data_latent_linear_listwise), loaded[0]['latent_linear'].shape[0],
                                   loaded[0]['latent_linear'].shape[1]])
    for i in range(len(data_latent_linear_listwise)):
        data_latent_linear[i] = data_latent_linear_listwise[i]

    data_latent_rnn_ndarray = np.array(data_latent_rnn)

    # Calculate the distance at the representation embedding space
    # calculate_distances(data_latent_rnn_ndarray)


    # scaler = StandardScaler()
    # scaled_features = scalar.fit_transform(data)

    # 1. clustering parameters.
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
        }

    if False:
        # 2. Analysis 1:
        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(data_latent_rnn)
            sse.append(kmeans.inertia_)

        plt.style.use("fivethirtyeight")
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()

    # 3. Analysis 2
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    # Notice you start at 2 clusters for silhouette coefficient
    if False:
        for k in range(50, 400, 25):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(data_latent_rnn)
            score = silhouette_score(data_latent_rnn, kmeans.labels_)
            silhouette_coefficients.append(score)
        plt.figure()
        plt.style.use("fivethirtyeight")
        plt.plot(range(50, 400, 25), silhouette_coefficients)
        plt.xticks(range(50, 400, 25))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")
        plt.show()

    k_component = 40

    if False:
        # 4. Dynamic Time Warping.
        # X = random_walks(n_ts=50, sz=32, d=1)
        # 4.1 Euclidean distance
        print("4.1 Euclidean distance")

        # print(data_latent_linear[0].shape)

        km = TimeSeriesKMeans(n_clusters=k_component, metric="euclidean", max_iter=5,
        random_state = 0).fit(data_latent_linear)
        km.cluster_centers_.shape
        labels = km.labels_


        # 4.1.1 save results
        print("4.1.1 save results")
        for i in tqdm(range(0, len(data_latent_linear), 10)):
            directory = str(save_result_path) + '/4.1/' + str(labels[i])
            if not os.path.exists(directory):
                os.makedirs(directory)
            make_bvh(directory, str(i), data_original[i])

    if False:
        # 4.2 DTW distance
        print(".2 DTW distance")
        km_dba = TimeSeriesKMeans(n_clusters=k_component, metric="dtw", max_iter=5,
                                  max_iter_barycenter=5,
                                  random_state=0).fit(data_latent_linear)
        km_dba.cluster_centers_.shape
        labels = km_dba.labels_

        # 4.2.1 save results
        print("4.2.1 save results")
        for i in tqdm(range(0, len(data_latent_linear), 10)):
            directory = str(save_result_path) + '/4.2/' + str(labels[i])
            if not os.path.exists(directory):
                os.makedirs(directory)
            make_bvh(directory, str(i), data_original[i])

    if False:
    # 4.3 softDTW distance
        print("4.3 softDTW distance")
        km_sdtw = TimeSeriesKMeans(n_clusters=k_component, metric="softdtw", max_iter=5,
                                    max_iter_barycenter = 5,
                                    metric_params = {"gamma": .5},
                                    random_state = 0).fit(data_latent_linear)
        km_sdtw.cluster_centers_.shape
        labels = km_sdtw.labels_

        # 4.3.1 save results
        print("4.3.1 save results")
        for i in tqdm(range(0, len(data_latent_linear), 10)):
            directory = str(save_result_path) + '/4.3/' + str(labels[i])
            if not os.path.exists(directory):
                os.makedirs(directory)
            make_bvh(directory, str(i), data_original[i])

    # 4.4 RNN based clustering
    print("4.4 RNN based")

    scaler = StandardScaler()
    data_latent_rnn_ndarray = data_latent_rnn_ndarray.squeeze()
    scaler.fit(data_latent_rnn_ndarray)
    # Todo: check which one is better? Unnormalized showed better results.
    # normalized_data = scaler.transform(data_latent_rnn_ndarray)
    normalized_data = data_latent_rnn_ndarray


    # 4.4.0 Algorithm
    Algorithm = 'kmeans'
    k_component = 300
    if Algorithm == 'kmeans':

        # Save address:
        km_address = (save_result_path) + '/'
        os.makedirs(km_address, exist_ok=True)
        km_address += 'kmeans_model.pk'

        if os.path.exists(km_address):
            km_rnn = pickle.load(open(km_address, 'rb'))
            print("Kmeans Loaded!")

        else: # train kmeans
            km_rnn = KMeans(n_clusters=k_component,  max_iter=2500,
                                       random_state=0).fit(normalized_data)
            print("Kmeans trained!")
            km_rnn.cluster_centers_.shape
            # Save model
            # km_rnn.to_pickle('../output/clustering_results/kmeans_model.pk')
            # pickle.dump(km_rnn, open('../output/clustering_results/kmeans_model.pk', 'wb'))
            pickle.dump(km_rnn, open(km_address, 'wb'))
        labels = km_rnn.labels_
        labels = list(labels)


    elif Algorithm == 'mdp':
        X = normalized_data.T
        N = X.shape[1]
        # Set up Normal-Wishart MAP-DP prior parameters
        N0 = 0.5  # Prior count (concentration parameter)
        m0 = X.mean(1)[:, None]  # Normal-Wishart prior mean
        a0 = 10  # Normal-Wishart prior scale
        c0 = 10 / float(N)  # Normal-Wishart prior degrees of freedom
        B0 = np.diag(1 / (0.05 * X.var(1)))  # Normal-Wishart prior precision
        # # Run MAPDP to convergence

        mu, labels, k_component, E = mapdp_nw(X, N0, m0, a0, c0, B0)
    #
    elif Algorithm == 'dbscan':
        db = DBSCAN(eps=0.05, min_samples=10).fit(normalized_data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        k_component = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
    elif Algorithm == 'AgglomerativeClustering':
        clustering  = AgglomerativeClustering(linkage='ward',
                                              n_clusters=k_component).fit(normalized_data)
        labels = clustering.labels_
    # 4.4.1 save results



    # Plot statistic
    objects = [str(x) for x in range(0, k_component)]
    y_pos = np.arange(len(objects))
    # performance = [labels.count(i) for i in labels]

    a = dict(Counter(labels))

    # Sanity check
    check4 = 0
    # for uuu in range(len(labels)):
    #     if labels[uuu] == 4:
    #         check4 = check4 +1
    # Passed

    plot_Portion = False
    if plot_Portion:
        total = 0
        # for saving the portion
        for index in a:
            total += a[index]
        portion = []
        for i in range(0, k_component):
            if i == -1:
                continue
            portion.append(a[i]/total)
        portion = np.array(portion)
        performance = []
        for i in range(0, k_component):
            performance.append(a[i])
        # plt.figure(figsize=(100,400))
        plt.bar(y_pos, performance, align='center', alpha=0.5, width=0.8)
        plt.xticks(y_pos, objects, size='small')
        plt.xticks(rotation=90)
        plt.ylabel('occurrence')
        plt.title('clustered samples')

        plt.show()


    print("4.4.1 save results")

    save = True
    # save = True

    # Save clusters' centers
    if False:
        if Algorithm =='kmeans':
            for c_i, clusters_centers in enumerate(km_rnn.cluster_centers_):
                directory = str(save_result_path) + '/4.4/' + str(c_i)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                min_dist_1 = np.sqrt(np.mean(
                    km_rnn.transform(
                        data_latent_rnn_ndarray[0, :].reshape(1, -1)
                    )[:, c_i]**2))
                min_index_1 = 0

                min_dist_2 = np.sqrt(np.mean(
                    km_rnn.transform(
                        data_latent_rnn_ndarray[1, :].reshape(1, -1)
                    )[:, c_i] ** 2))
                min_index_2 = 1

                if min_dist_1 > min_dist_2:
                    min_dist_1, min_dist_2 = min_dist_2, min_dist_1
                    min_index_1, min_index_2 = min_index_2, min_index_1

                for s_i in tqdm(range(0, len(data_latent_linear))):

                    # check if the instance belongs to c_index cluster
                    if labels[s_i] != c_i:
                        continue

                    # array.reshape(1, -1) if it contains a single sample
                    reshaped = data_latent_rnn_ndarray[s_i, :]
                    reshaped = reshaped.reshape(1, -1)
                    current_distance = km_rnn.transform(reshaped)[:, c_i]
                    current_distance = np.mean(current_distance**2)
                    current_distance = np.sqrt(current_distance)
                    if current_distance < min_dist_1:
                        min_index_2 = min_dist_1
                        min_index_2 = min_index_1
                        min_dist_1 = current_distance
                        min_index_1 = s_i
                    elif current_distance < min_dist_2:
                        min_dist_2 = current_distance
                        min_index_2 = s_i
                if DATA_TYPE=='Trinity':
                    make_bvh(directory, "Closest1-" + str(min_index_1), data_original[min_index_1])
                    make_bvh(directory, "Closest2-" + str(min_index_2), data_original[min_index_2])
                elif DATA_TYPE=='TWH':
                    make_bvh_Trinity(directory, "Closest1-" + str(min_index_1), data_original[min_index_1])
                    make_bvh_Trinity(directory, "Closest2-" + str(min_index_2), data_original[min_index_2])


    if True:
        if Flag_VQ:
            for i in tqdm(range(0, 7000, 1)):
                directory = str(save_result_path) + '/VQ_clusters/' + str(data_quantized_indicies[i])
                if not os.path.exists(directory):
                    os.makedirs(directory)
                if DATA_TYPE=='Trinity':
                    make_bvh(directory, str(i), data_original[i])
                elif DATA_TYPE=='TWH':
                    make_bvh_Trinity(directory, str(i), data_original[i])
        # for i in tqdm(range(0, len(data_latent_linear), 1)):
        for i in tqdm(range(0, 4000, 1)):
            directory = str(save_result_path) + '/kmeans_clusters/' + str(labels[i])
            if not os.path.exists(directory):
                os.makedirs(directory)
            if DATA_TYPE=='Trinity':
                make_bvh(directory, str(i), data_original[i])
            elif DATA_TYPE=='TWH':
                make_bvh_Trinity(directory, str(i), data_original[i])


    # 5. Here I will implent flowgmm based clustering
    # todo: implement it.



    # n. Visualization when?
    # Prepare for clustrting visualization:
    directory = str(save_result_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    portion = None
    if True:
        if Flag_VQ:
            plot_tsne(args, normalized_data, scaler, k_component, np.array(data_quantized_indicies),
                      directory + '/transforms.pkl', True, portion)
        else:
            plot_tsne(args, normalized_data, scaler, k_component, km_rnn.labels_,
                      directory + '/transforms.pkl', True, portion)

    # Histogram_clusters(args, km_rnn, normalized_data, scaler, k_component, km_rnn.labels_, '../output/transforms.pkl', False, portion)
    # Metrics_analysis(args, km_rnn, normalized_data, scaler,
    #                  k_component, km_rnn.labels_, '../output/transforms.pkl', False)


    # 3D
    exit()
    X_embedded = TSNE(n_components=3).fit_transform(normalized_data)

    sns.set(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_zlabel("Dim3")
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=km_rnn.labels_)
    plt.show()
    exit()
    pickle.dump(kmeans.labels_, open('../output/kmeans_labels.bin', 'wb'))

    # # DECODE GESTURES BASED ON CLUSTERS
    # for i in range(len(data)):
    #     current_data = data[i]
    #     current_data = torch.from_numpy(current_data).to(device)
    #     decoder_hidden = current_data


    exit()



    # Compute DBSCAN

    db = DBSCAN(eps=eps, min_samples=10).fit(scaled_features)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    k_component = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result


    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def plot_tsne(args, normalized_data, scalar, k_component, labels_, save_adress, rewrite, portion):


    if rewrite:
        # PCA
        pca = PCA(n_components=50)
        priciple_components = pca.fit(normalized_data)
        normalized_data = priciple_components.transform(normalized_data)
        # TSNE

        MyTSNE = TSNE(
            n_components=2,
            perplexity=30,
            metric='euclidean',
            n_jobs=8,
            verbose=True
        )
        X_embedded = MyTSNE.fit(normalized_data)

        #     Save
        stats = {'pca': pca, 'tsne': X_embedded, 'scalar': scalar, 'portion': portion}
        pickle.dump(stats, open(save_adress, 'wb'))
    else:
        # PCA
        loaded = pickle.load(open(save_adress, 'rb'))
        pca, MyTSNE = loaded['pca'], loaded['tsne']
        normalized_data = pca.transform(normalized_data)
        # TSNE
        X_embedded = MyTSNE.transform(normalized_data)


    plt.figure(figsize=(16, 10))
    palette = sns.color_palette("Set2", k_component)
    # 2D
    min = np.min(labels_)
    max = np.max(labels_)
    a_set = set(labels_)
    number_of_unique_values = len(a_set)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1],
                    hue=labels_, legend=False, palette=sns.color_palette('Set2', n_colors=number_of_unique_values))

    address = os.path.dirname(args.ckpt_path_Autoencode) + '/TSNE.png'
    plt.savefig(address)
    plt.show()



    # Hitmap
    x_lim, y_lim = 60, 60

    heatmap, x_edge, y_edge = np.histogram2d(X_embedded[:, 0], X_embedded[:, 1], bins=100)
    extent = [x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]]
    extent = [-y_lim, y_lim, -x_lim, x_lim]
    plt.clf()
    sigma = 2
    for sigma in range(2, 3):
        plt.figure(figsize=(16, 10))
        heatmap_filtered = gaussian_filter(heatmap, sigma)
        plt.imshow(heatmap_filtered.T, extent=extent, origin='lower')
        # plt.colorbar(plt.pcolor(heatmap))
        # plt.pcolor(heatmap)
        plt.colorbar()
        plt.title('Heatmap of representation dataset, Sigma = ' + str(sigma))

        address = os.path.dirname(args.ckpt_path_Autoencode) + '/Hitmap.png'
        plt.savefig(address)

        plt.show()
        plt.clf()


    return

    # # Plot generated
    bvh_file1 = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/Co-Speech_Gesture_Generation/output' \
               '/Experiment/Objective/BVH_submitted/Ground_Truth/TestSeq001.bvh'
    bvh_file2 = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/Co-Speech_Gesture_Generation/output' \
               '/Experiment/Objective/BVH_submitted/Cond_BA/TestSeq001.bvh'
    bvh_file3 = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/Co-Speech_Gesture_Generation/output' \
               '/Experiment/Objective/BVH_submitted/Cond_BT/TestSeq001.bvh'
    bvh_file4 = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/Co-Speech_Gesture_Generation/output' \
               '/Experiment/Objective/BVH_submitted/Cond_SD/TestSeq001.bvh'
    bvh_file5 = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/Co-Speech_Gesture_Generation/output' \
               '/Experiment/Objective/BVH_submitted/proposed_not_smooth/TestSeq001.bvh'

    bvh_files = [bvh_file1, bvh_file2, bvh_file3, bvh_file4, bvh_file5]
    for bvh_file in bvh_files:
        latent_org = maake_dataset_for_inference(args.ckpt_path_DAE, args.ckpt_path_Autoencode, bvh_file)
        all_latents = []
        for item in latent_org:
            current_rnn = item['latent_rnn']
            current_rnn = np.hstack(current_rnn)
            all_latents.append(current_rnn)
        all_latents = np.array(all_latents)
        all_latents = scalar.transform(all_latents) #normalization
        all_latents = pca.transform(all_latents)
        all_latents = MyTSNE.transform(all_latents)
        plt.scatter(all_latents[:, 0], all_latents[:, 1], c='r')
        plt.xlim(-y_lim, y_lim)
        plt.ylim(-x_lim, x_lim)
        plt.title(bvh_file[-30:-15])
        plt.show()
        # Heatmap of it
        heatmap, x_edge, y_edge = np.histogram2d(all_latents[:, 0], all_latents[:, 1], bins=100)
        extent = [x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]]
        extent = [-y_lim, y_lim, -x_lim, x_lim]
        plt.clf()
        sigma = 1
        heatmap_filtered = gaussian_filter(heatmap, sigma)
        plt.imshow(heatmap_filtered.T, extent=extent, origin='lower')
        plt.colorbar()
        plt.title('Heatmap of generated infer, Sigma = ' + str(sigma))
        plt.xlabel(bvh_file[-20:])
        plt.show()
        plt.clf()



def Histogram_clusters(args, km, normalized_data, scalar, k_component, labels_, save_adress, rewrite, portion):

    from sklearn.decomposition import PCA
    from openTSNE import TSNE
    import seaborn as sns
    from scipy.ndimage.filters import gaussian_filter


    # # Plot generated
    dir_submitted = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/Co-Speech_Gesture_Generation/output' \
               '/Experiment/Objective/BVH_submitted/'
    conditions = ['Ground_Truth', 'Cond_BA', 'Cond_BT', 'Cond_SD',
                  'proposed_not_smooth', 'proposed_smooth_long',
                  'proposed_14']
    Hist_GT = None
    str_out = ''

    for cond_dir in conditions:
        bvh_files = sorted(glob.glob(os.path.join(dir_submitted + cond_dir, '*.bvh')))
        all_latents = []
        for bvh_file in bvh_files:
            latent_org = maake_dataset_for_inference(args.ckpt_path_DAE, args.ckpt_path_Autoencode, bvh_file)
            for item in latent_org:
                current_rnn = item['latent_rnn']
                current_rnn = np.hstack(current_rnn)
                all_latents.append(current_rnn)

        all_latents = np.array(all_latents)
        all_latents = scalar.transform(all_latents) #normalization

        labels = km.predict(all_latents)
        labels_dict = dict(Counter(labels))
        labels_x = range(0,k_component)
        labels_y = np.zeros(k_component)
        for i in range(0,300):
            if i in labels_dict:
                labels_y[i] = labels_dict[i]
        plt.plot(labels_x, labels_y)
        plt.title(bvh_file[-30:-15])
        plt.show()

        # todo: move it up
        if bvh_file.__contains__('Ground_Truth'):
            Hist_GT = labels_y

        hell_dist = hellinger(Hist_GT, labels_y)
        str_out += ("* " + bvh_file[-30:-15] + "-->" +  str(hell_dist)) + '\n'
        # str_out += ("* " + os.path.dirname(bvh_file) + "-->" + str(hell_dist)) + '\n'



        print(str_out)




def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    print('&&&&&&', mu1.shape, mu2.shape)
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def make_unity_scatter(this_seq_latent, this_seq_labels, save_address, pca, MyTSNE):

    normalized_data = pca.transform(this_seq_latent)
    # TSNE
    X_embedded = MyTSNE.transform(normalized_data)



    # str_out = 'Cluster_index, x_TSNE, y_TSNE\n'
    str_out = '512\n'
    for i in range(len(this_seq_labels)):
        str_out += str(this_seq_labels[i]) + ',{:.3f},'.format(X_embedded[i,0]) + '{:.3f}\n'.format(X_embedded[i,1])

    file1 = open(save_address, 'w')
    file1.write(str_out)
    file1.flush()
    file1.close()

def Metrics_analysis(args, km, normalized_data, scalar,
                     k_component, labels_, save_adress, rewrite):



    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist

    def wasserstein_distance_calc(portion_A, portian_B):
        # u = [0.5, 0.2, 0.3]
        # v = [0.5, 0.3, 0.2]
        # create and array with cardinality 3 (your metric space is 3-dimensional and
        # where distance between each pair of adjacent elements is 1
        dists = [i for i in range(len(portion_A))]

        return stats.wasserstein_distance(dists, dists, portion_A, portian_B)

    # frechet distance
    # frechet_dist = frechet_distance(generated_feats, real_feats)

    # def features_distance():
    #     # distance between real and generated samples on the latent feature space
    #     dists = []
    #     for i in range(real_feats.shape[0]):
    #         d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
    #         dists.append(d)
    #     feat_dist = np.mean(dists)

    import contextlib

    # ____________________________
    # Train TSNE Transformation
    pca = PCA(n_components=50)
    priciple_components = pca.fit(normalized_data)
    normalized_data = priciple_components.transform(normalized_data)
    MyTSNE = TSNE(
        n_components=2,
        perplexity=30,
        metric='euclidean',
        n_jobs=8,
        verbose=True
    )
    MyTSNE = MyTSNE.fit(normalized_data)
    # ____________________________


    # # Plot generated
    dir_submitted = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/Co-Speech_Gesture_Generation/output' \
               '/IROS_2/BVH_submitted/'
    conditions = ['Ground_Truthx',
                  'Cond_BA', 'Cond_BT', 'Cond_SD',

                  'infer_sample_1', 'infer_sample_2', 'infer_sample_3', 'infer_sample_4', 'infer_sample_5'

                                                                                        ,'infer_sample_1r',
                  'infer_sample_2r', 'infer_sample_3r', 'infer_sample_4r', 'infer_sample_5r',
                  'infer_sample_6r', 'infer_sample_7r', 'infer_sample_8r', 'infer_sample_9r', 'infer_sample_9r_2',
                  'infer_sample_10r'

                  # 'infer_sample_2_L', 'infer_sample_3_L', 'infer_sample_4_L', 'infer_sample_5_L',
                  # 'infer_sample_6_L', 'infer_sample_7_L', 'infer_sample_8_L', 'infer_sample_9_L',
                  # 'infer_sample_10_L', 'infer_sample_11_L', 'infer_sample_12_L', 'infer_sample_13_L',
                  # 'infer_sample_16_L', 'infer_sample_16_r0', 'infer_sample_16_r1', 'infer_sample_17_Lr0',
                  # 'infer_sample_2', 'infer_sample_3', 'infer_sample_4', 'infer_sample_4_2',
                  # 'infer_sample_5', 'infer_sample_6', 'infer_sample_7', 'infer_sample_7_2', 'infer_sample_7_3',
                  # 'infer_sample_7_4', 'infer_sample_7_5',
                  # 'infer_sample_8', 'infer_sample_9', 'infer_sample_10',
                  # 'infer_sample_11', 'infer_sample_12', 'infer_sample_13',
                  ]
    Hist_GT = None
    str_out = ''
    str_latex_ablation = ''
    str_latex = ''
    GT_latents = None
    GT_pdf = []
    sequences = dict()
    for cond_dir in tqdm(conditions):
        if cond_dir == 'proposed_10':
            print("Wait!")
        bvh_files = sorted(glob.glob(os.path.join(dir_submitted + cond_dir, '*.bvh')))
        all_latents = []
        all_labels = []
        sequences[cond_dir] = dict()
        for bvh_file in tqdm(bvh_files):

            testcase_id = bvh_file[-7:-4]
            this_seq_latent = []
            this_seq_labels = [] # for K-Means algorithm

            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                latent_org = maake_dataset_for_inference(args.ckpt_path_DAE,
                                                         args.ckpt_path_Autoencode, bvh_file)


            # labels = np.zeros(len(latent_org), dtype=int)
            for i, item in tqdm(enumerate(latent_org)):
                # labels[i] = int(latent_org[i]['quantized_indices'])
                # Store_latent
                current_rnn = item['latent_rnn']
                current_rnn = np.hstack(current_rnn)
                this_seq_latent.append(current_rnn)


                # Store labels
                if 'quantized_indices' in latent_org[0] and False:
                    current_quantized_label = int(latent_org[i]['quantized_indices'])
                    this_seq_labels.append(current_quantized_label)
                    # all_labels.append(current_quantized_label)

            if 'quantized_indices' not in latent_org[0] or True:
                to_predict = np.array(this_seq_latent)
                this_seq_labels = km.predict(to_predict)

            make_unity_scatter(np.array(this_seq_latent), this_seq_labels, (bvh_file[:-4] + '_unuryscatter.txt'),
                               pca, MyTSNE)



        # all_latents = np.array(all_latents)
        # to_predict = np.array(to_predict)
        # all_latents = scalar.transform(all_latents) #normalization

        # labels = km.predict(all_latents)

            # Store for BLEU
            sequences[cond_dir][testcase_id] = np.array(this_seq_labels)
            all_latents.extend(this_seq_latent)
            all_labels.extend(this_seq_labels)
            # if testcase_id == '002':



        # all_latents = np.hstack(all_latents)
        # all_labels = np.vstack(all_labels)

        labels_dict = dict(Counter(all_labels))
        # k_component = 512
        labels_x = range(0, k_component)
        labels_y = np.zeros(k_component)
        for i in range(0,k_component):
            if i in labels_dict:
                labels_y[i] = labels_dict[i]
        plt.plot(labels_x, labels_y)
        plt.title(bvh_file[-30:-15])
        plt.show()

        # todo: move it up
        if bvh_file.__contains__('Ground_Truth'):
            Hist_GT = labels_y
            GT_latents = all_latents
            GT_pdf = labels_y / np.sum(labels_y)
            ref_sequences_id = cond_dir


        labels_y_pdf = labels_y/np.sum(labels_y)
        ppl = np.exp( - np.sum(labels_y_pdf*np.log(labels_y_pdf++ 1e-10)   ) )


        # 1. Hellinger Distance
        hell_dist = hellinger(Hist_GT, labels_y)
        str_out += '\n' + cond_dir + '\n'
        str_latex_ablation
        str_out += '\n Perplexity: ' + str(ppl) + '\n'
        str_out += ('hell_dist' + " --> " + str(hell_dist)) + '\n'
        # str_out += ("* " + os.path.dirname(bvh_file) + "-->" + str(hell_dist)) + '\n'

        # 2.
        frech_dist = frechet_distance(GT_latents, all_latents)
        str_out += 'Frechet Distance --> ' + str(frech_dist) + '\n'

        # 3.
        wasserstein_distance = wasserstein_distance_calc(GT_pdf, labels_y / np.sum(labels_y))
        str_out+='wasserstein_distance -> ' + str(wasserstein_distance)

        # 4. BLEU
        from torchtext.data.metrics import bleu_score
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.bleu_score import corpus_bleu
        pose_sentence_length = 4
        current_cond_score_list = []
        for test_case in sequences[cond_dir]:
            predicted_corpus = sequences[cond_dir][test_case]
            references_corpus = sequences[ref_sequences_id][test_case]

            predicted_sentence_all = []
            references_sentence_all = []




            predicted_sentence = predicted_corpus[:] #[word_i: word_i + pose_sentence_length]
            predicted_sentence = predicted_sentence.tolist()
            predicted_sentence = list(map(str, predicted_sentence))
            # predicted_sentence_all.append(predicted_sentence)
            predicted_sentence = [predicted_sentence]


            references_sentence = references_corpus[:] #[word_i: word_i + pose_sentence_length]
            references_sentence = references_sentence.tolist()
            references_sentence = list(map(str, references_sentence))
            # references_sentence_all.append((references_sentence))
            references_sentence = [[references_sentence]]

            score = bleu_score(predicted_sentence, references_sentence, max_n=4,
                               weights=[0.25, 0.25, 0.25, 0.25])
            # score = sentence_bleu(references_sentence, predicted_sentence)
            current_cond_score_list.append(score)


        avg_BLEU_Score = np.mean(current_cond_score_list)
        std_BLEU_Score = np.var(current_cond_score_list)
        str_out += "\n BLEU: "+ str(np.sum(current_cond_score_list))+" AVG_BLEU Score:" + str(avg_BLEU_Score) + " +-" + str(std_BLEU_Score) + "\n\n\n"
        print(str_out)
        str_latex += cond_dir.replace('_', '-') + ' & ' + '${:.3f}$'.format(frech_dist) + ' & ' + '${:.3f}$'.format(hell_dist) + \
        ' & ' + '${:.2f}$'.format(ppl) + ' & ' + '${:.3f}$'.format(avg_BLEU_Score) + ' \\\\ \n'




    metric_file = open(save_result_path + "/Metrics.txt", 'w')
    metric_file.write(str_out + '\n\nLatex\n' + str_latex)
    metric_file.flush()
    metric_file.close()
    exit()


def normalize(hist):
    return hist / np.sum(hist)


def hellinger(hist1, hist2):
    """Compute Hellinger distance between two histograms

      Args:
          hist1:        first histogram
          hist2:        second histogram of the same size as hist1

      Returns:
          float:        Hellinger distance between hist1 and hist2
    """

    return np.sqrt(1.0 - np.sum(np.sqrt(normalize(hist1) * normalize(hist2))))

def FGD():
    pass

# Todo: Remove this from here???
# Todo:-------------------------------------------------------
from scipy.special import gammaln
from scipy import linalg
import numpy as np
from numpy.matlib import repmat
# from numpy.linalg import inv, linalg


def mapdp_nw(X, N0, m0, a0, c0, B0, epsilon=1e-6, maxiter=100, fDebug=False):
    '''
    MAP-DP for Normal-Wishart data.

    Inputs:  X  - DxN matrix of data
             N0 - prior count (DP concentration parameter)
             m0 - cluster prior mean
             a0 - cluster prior scale
             c0 - cluster prior degrees of freedom
             B0 - cluster prior precision (inverse covariance)
             epsilon - convergence tolerance
             maxiter - maximum number of iterations, overrides threshold calculation
             fDebug - extra verbose input of algorithm operation

    Outputs: mu - cluster centroids
             z  - data point cluster assignments
             K  - number of clusters
             E  - objective function value for each iteration

    CC BY-SA 3.0 Attribution-Sharealike 3.0, Max A. Little. If you use this
    code in your research, please cite:
    Yordan P. Raykov, Alexis Boukouvalas, Fahd Baig, Max A. Little (2016)
    "What to do when K-means clustering fails: a simple yet principled alternative algorithm",
    PLoS One, (11)9:e0162259
    This implementation follows the description in that paper.
    '''

    # Initialization (Alg. 3 line 1)
    (D, N) = X.shape
    assert (D > 0)
    assert (N > 0)
    K = 1
    z = np.zeros((N), dtype=int)  # everybody assigned to first cluster
    Enew = np.inf
    dE = np.inf
    ic = 0  # iteration coung
    E = list()
    # Convergence test (Alg. 3 line 14 and Appendix B)
    while (abs(dE) > epsilon and ic < maxiter):
        print("Iteration:", ic)
        Eold = Enew
        dik = np.ones((N, 1)) * np.inf
        for i in range(N):
            dk = np.ones((K + 1, 1)) * np.inf
            f = np.empty(K + 1)
            Nki = np.ones((K), dtype=int)
            xi = np.atleast_2d(X[:, i]).T  # current data point
            for k in range(K):
                zki = (z == k)
                zki[i] = False
                Nki[k] = zki.sum()
                # Updates meaningless for Nki=0
                if (Nki[k] == 0):
                    continue
                # Update NW cluster hyper parameters (Alg. 3 line 7)
                mki, aki, cki, Bki = nwupd(Nki[k], X[:, zki], m0, a0, c0, B0)

                # Compute Student-t NLL, existing clusters (Alg. 3 line 8)
                dk[k] = stnll(xi, mki, aki, cki, Bki, D)
                # Avoid reinforcement effect at initialization (Appendix B)
                if (ic == 0):
                    Nki[0] = 1
                f[k] = dk[k] - np.log(Nki[k])
            # Compute Student-t NLL, new cluster (Alg. 3 line 9)
            dk[K] = stnll(xi, m0, a0, c0, B0, D)
            f[K] = dk[K] - np.log(N0)
            # Compute MAP assignment (Alg. 3 line 10)
            if (fDebug):
                print(i, 'Compute MAP assignment K=', K, 'f=', f, 'dk=', dk)

            z[i] = np.argmin(f)
            dik[i] = f[z[i]]
            # Create new cluster if required (Alg. 3 line 11-12)
            if (z[i] == K):
                K = K + 1
        # Remove any empty clusters and re-assign (Appendix B)
        Knz = 0
        for k in range(K):
            i = (z == k)
            Nk = i.sum()
            if (Nk > 0):
                z[i] = Knz
                Knz = Knz + 1
        K = Knz
        Nk, _ = np.histogram(z, range(K + 1))
        # Compute updated NLL (Alg. 3 line 13)
        Enew = dik.sum() - K * np.log(N0) - np.sum(gammaln(Nk))
        dE = Eold - Enew
        ic += 1
        E.append(Enew)
        print('Iteration %d: K=%d, E=%f, dE=%f\n' % (ic, K, Enew, dE))

    # Compute cluster centroids (Appendix D)
    mu = np.ones((D, K))
    for k in range(K):
        xk = X[:, z == k]
        mu[:, k] = xk.mean(1)
    return mu, z, K, E


def stnll(x, m, a, c, B, D):
    '''
    Compute Student-t negative log likelihood (Appendix A, eqn. (20))
    '''
    mu = m
    nu = a - D + 1
    Lambda = c * float(nu) / (c + 1) * B
    S = np.dot(np.dot((x - mu).T, Lambda), (x - mu))
    _, logdetL = slogdet(Lambda)
    return float(nu + D) / 2. * np.log(1. + S / float(nu)) \
           - 0.5 * logdetL + gammaln(nu / 2.) \
           - gammaln((float(nu) + D) / 2.) + D / 2. * np.log(float(nu) * np.pi)


def nwupd(Nki, xki, m0, a0, c0, B0):
    '''
    Update Normal-Wishart hyper parameters (Appendix A, eqns. (18-19))
    '''
    xmki = xki.mean(1)[:, None]
    xmcki = xki - repmat(xmki, 1, Nki)
    Ski = np.dot(xmcki, xmcki.T)
    cki = c0 + Nki
    mki = (c0 * m0 + Nki * xmki) / cki
    xm0cki = xmki - m0
    Bki = inv(inv(B0) + Ski + c0 * Nki / cki * np.dot(xm0cki, xm0cki.T))
    aki = a0 + Nki
    return mki, aki, cki, Bki


# Todo:-------------------------------------------------------

def test_mapdp():
    d = pd.read_csv('toydata.csv')
    N = d.shape[0]
    X = d[['X_1', 'X_2']].values.T
    Z = d['Z'].values
    # Set up Normal-Wishart MAP-DP prior parameters
    N0 = 0.5  # Prior count (concentration parameter)
    m0 = X.mean(1)[:, None]  # Normal-Wishart prior mean
    a0 = 10  # Normal-Wishart prior scale
    c0 = 10 / float(N)  # Normal-Wishart prior degrees of freedom
    B0 = np.diag(1 / (0.05 * X.var(1)))  # Normal-Wishart prior precision
    # # Run MAPDP to convergence
    mu, z, K, E = mapdp_nw(X, N0, m0, a0, c0, B0)
    # # Plotting
    plt.figure(figsize=(10, 5))
    for j in range(1, K + 1):
        i = (Z == j);
        plt.plot(X[0, i], X[1, i], '.')
    plt.title('Ground truth')

    plt.figure(figsize=(10, 5))
    for j in range(K):
        i = (z == j);
        plt.plot(X[0, i], X[1, i], '.')
    plt.title('MAPDP estimated clusters')

    plt.figure(figsize=(10, 5))
    plt.plot(E, '-b')
    plt.title('MAPDP objective function')
    plt.show()
    exit()


if __name__ == '__main__':

    '''
   ../output/DAE_old/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
../output/autoencoder/toturial/ICLR_VQ_DVAE/VQ-DVAE_ablation1_checkpoint_020.bin


 ../output/GENEA/DAE/train_DAE_H45/DAE_H45_checkpoint_020.bin
../output/GENEA/VQ-VAE/VQVAE_checkpoint_020.bin
    '''


    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path_DAE", type=Path)
    parser.add_argument("ckpt_path_Autoencode", type=Path)
    # parser.add_argument("pre_processed_pickle", type=Path)
    # parser.add_argument("save_result_path", type=str)
    args = parser.parse_args()

    args.save_result_path = os.path.dirname(args.ckpt_path_Autoencode) + '/clusters'
    if not os.path.exists(args.save_result_path):
        os.makedirs(args.save_result_path)

    args.pre_processed_pickle = args.save_result_path + '/org_latent_clustering_data.bin'

    if not os.path.exists(args.pre_processed_pickle):
        maake_dataset(args.ckpt_path_DAE, args.ckpt_path_Autoencode,
                      args.save_result_path + '/org_latent_clustering_data.bin')

    # if Flag_VQ:
    #     make_VQ_Centers(args.ckpt_path_DAE, args.ckpt_path_Autoencode,
    #                       args.save_result_path)

    loaded = pickle.load(open(args.pre_processed_pickle,'rb'))
    liaded_len = len(loaded)
    loaded = np.hstack(loaded)
    print('len loaded', len(loaded))
    print("Loaded successfully")
    eps = 3
    min_samples = 5
    save_result_path = args.save_result_path
    clustered = cluster(loaded, eps, min_samples, save_result_path)