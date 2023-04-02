"""
"""


import logging
import os
import pickle
import random
from typing import Tuple

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from data_loader.data_preprocessor import DataPreprocessor
import pyarrow
from configargparse import argparse

from model.vocab import Vocab
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def word_seq_collate_fn(data: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Collate function for loading word sequences in variable lengths.

    Args:
        data: A list of samples

    Returns:
        A 5-Tuple or 8-Tuple:

        8-Tuple (if sentence_level is set to True):
            word_seq:
            words_lengths:
            poses_seq:
            audio:
            aux_info:
            sentence_leve_latents:
            cluster_portion:
            GPT3_Embedding:

        5-Tuple:
            word_seq:
            words_lengths:
            poses_seq:
            audio:
            aux_info:
    """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)


    # separate source and target sequences
    # Todo: fix this using args
    sentence_leve = True
    if not sentence_leve:
        word_seq, poses_seq, audio, aux_info, = zip(*data)
    else:
        word_seq, poses_seq, audio, aux_info, sentence_leve_latents, cluster_portion, GPT3_Embedding = zip(*data)




    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    poses_seq = default_collate(poses_seq)
    audio = default_collate(audio)
    if sentence_leve:
        sentence_leve_latents = default_collate(sentence_leve_latents)
        cluster_portion = default_collate(cluster_portion)
        GPT3_Embedding = default_collate(GPT3_Embedding)

    # audio = default_collate(audio)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    if sentence_leve:
        return word_seq, words_lengths, poses_seq, audio, aux_info, sentence_leve_latents, cluster_portion, GPT3_Embedding
    else:
        return word_seq, words_lengths, poses_seq, audio, aux_info


class TrinityDataset(Dataset):
    """Contains information and and associated parameters of a (Trinity) dataset.

    This is a PyTorch Dataset subclass containing information of a Trinity dataset.
    (ex. https://trinityspeechgesture.scss.tcd.ie/data/Trinity%20Speech-Gesture%20I/GENEA_Challenge_2020_data_release/)

    Attributes:
        lmdb_dir: A string representing the filepath of the directory containing the actual dataset.
        lmdb_env: A Lmdb ('Lightning Memory-Mapped' Database) object loaded from .mdb files located at the lmdb_dir.
        n_poses: An int representing the number of frames in each clip in the dataset (normally 30 (in 30 fps)).
        subdivision_stride: An int representing the number of frames between the start of one clip and the start of the next clip (clips can overlap).
        skeleton_resampling_fps: An int representing the frames per second of clip to use for training (usually downsampled to 20 fps, clips are normally 30 fps).
        n_samples: An int representing the number of clips/entries in the original dataset.
        lang_model: A pre-trained (English) language vector representation contained in the 'Vocab' custom class.
        data_mean: A mean calculated from each video in the original dataset.
        data_std: A standard deviation calculcated from each video in the original dataset.
    """
    def __init__(self, args: argparse.Namespace, lmdb_dir: str, n_poses: int, subdivision_stride: int, pose_resampling_fps: int, data_mean: list[float], data_std: list[float]):
        """Initialization function.

        Args:
            args: A configargparse object containing all parameters for the dataset and training found in a specified config file (ex. config/DAE.yml)
            lmdb_dir: A string representing the filepath of the directory containing the actual dataset.
            n_poses: An int representing the number of frames in each clip in the dataset (normally 30 (in 30 fps)).
            subdivision_stride: An int representing the number of frames between the start of one clip and the start of the next clip (clips can overlap).
            pose_resampling_fps: An int representing the frames per second of clip to use for training (usually downsampled to 20 fps, clips are normally 30 fps).
            data_mean: A mean calculated from each video in the original dataset.
            data_std: A standard deviation calculcated from each video in the original dataset.
        """
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(args, lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps, sentence_level=False)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self) -> int:
        """Overridden length function.

        Returns:
            The number of videos/samples in the original dataset.
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Overridden getitem function.

        Retrieves a specific entry in the original dataset and associated information of that entry such as words, audio, poses, etc.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A 4-Tuple:
                word_seq_tensor: A Tensor of word vector representations.
                pose_seq: A Tensor of pose data.
                audio: A Tensor of audio data.
                aux_info: A dict containing information such as name, start/end frame info and start/end time.
        """
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, audio, aux_info = sample

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        pose_seq: torch.Tensor = (pose_seq - self.data_mean) / std

        # to tensors
        word_seq_tensor = words_to_tensor(self.lang_model, word_seq, aux_info['end_time'])
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(audio).float()

        return word_seq_tensor, pose_seq, audio, aux_info

    def set_lang_model(self, lang_model: Vocab) -> None:
        """Set the language vector representation to be used with the dataset.

        Modifies the internal state of the object.

        Args:
            lang_model: A pre-trained language vector representation contained in the 'Vocab' class.
        """
        self.lang_model = lang_model

class TrinityDataset_DAE(Dataset):
    def __init__(self, args, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):

            data_sampler = DataPreprocessor(args, lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']


        # Initiate all poses
        self.all_poses = []
        self.create_all_poses()
        print("data init finished!")


    def __len__(self):
        # return 500
        # return (self.n_samples * self.n_poses) //5
        return self.n_samples

    def create_all_poses(self):
        with self.lmdb_env.begin(write=False) as txn:
            for i in range(self.n_samples):
                key = '{:010}'.format(i).encode('ascii')
                sample = txn.get(key)

                sample = pyarrow.deserialize(sample)
                word_seq, pose_seq, audio, aux_info = sample

                # normalize
                std = np.clip(self.data_std, a_min=0.01, a_max=None)
                pose_seq = (pose_seq - self.data_mean) / std

                for j in range(0, len(pose_seq)):
                    original = pose_seq[j,:]
                    var_coef = 1 #std
                    sigma = 1
                    # noisy = self.add_noise(original, 0.0, std)
                    noisy = original # dropout layer adds noise instead
                    self.all_poses.append({'original': original, "noisy": noisy})


    def get_item_Memory_Efficient(self, idx):
        idx_lmdb = idx//self.n_poses

        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx_lmdb).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, audio, aux_info = sample


            # normalize
            std = np.clip(self.data_std, a_min=0.01, a_max=None)
            pose_seq = (pose_seq - self.data_mean) / std

            original = pose_seq[idx % self.n_poses, :]
            noisy = original

            original = torch.from_numpy(original).reshape((original.shape[0], -1)).float()
            noisy = torch.from_numpy(noisy).reshape((noisy.shape[0], -1)).float()

            return noisy, original

    def add_noise(self, x, variance_multiplier, sigma):
        """
               Add Gaussian noise to the data
               Args:
                   x                   - input vector
                   variance_multiplier - coefficient to multiple variance of the noise on
                   sigma               - variance of the dataset
               Returns:
                   x - output vector, noisy data
        """
        eps = 1e-15
        noise = np.random.normal(0.0, np.multiply(sigma, variance_multiplier) + eps, x.shape)
        x = x + noise
        return x
    def add_noise2(self, x, prob):
        """
               Add Gaussian noise to the data
               Args:
                   x                   - input vector
                   variance_multiplier - coefficient to multiple variance of the noise on
                   sigma               - variance of the dataset
               Returns:
                   x - output vector, noisy data
        """

        for i in range(len(x)):
            rnd = random.random()
            if (rnd < prob):
                x[i] = 0
        return x

    def __getitem__(self, idx):

        # Keep and document for only large datasets
        # Todo: I think this way is more efficient than the prev. approach
        # I need to double check and make sure everything works as before.
        # return self.get_item_Memory_Efficient(idx)
        # original, noisy = self.get_item_Memory_Efficient(idx)
        # return noisy, original


        original = self.all_poses[idx]['original']
        noisy = self.all_poses[idx]['noisy']

        original = torch.from_numpy(original).reshape((original.shape[0], -1)).float()
        noisy = torch.from_numpy(noisy).reshape((noisy.shape[0], -1)).float()

        return noisy, original

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model


class TrinityDataset_DAEed_Autoencoder(Dataset):
    def __init__(self, args, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()

        # We will change it to true once we call the creat_similarity_dataset
        self.pairwise_enabeled = False
        self.use_drivitive = args.use_derivitive == 'True'

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(args, lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    #     Todo: we need to initiate pre-trained representation learning model
        checkpoint_path = args.rep_learning_checkpoint
        self.rep_learning_dim = args.rep_learning_dim
        rep_learning_args, rep_model, rep_loss_fn, rep_lang_model, rep_out_dim = utils.train_utils.load_checkpoint_and_model(
            checkpoint_path, device, 'DAE')
        self.rep_model = rep_model.to('cpu')
        self.rep_model.train(False)

    def __len__(self):
        return self.n_samples//10

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, audio, aux_info = sample

        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        pose_seq = (pose_seq - self.data_mean) / std


        # Todo: Here we should apply rep_learning
        target = torch.from_numpy(pose_seq)
        # target = torch.unsqueeze(target, 2)
        target = target.float()
        # target = target.to(device)
        with torch.no_grad():
            if self.rep_model.encoder == None: #for ablation study
                encoded_poses = target
            else:
                encoded_poses = self.rep_model.encoder(target)

        # encoded_poses = torch.squeeze(encoded_poses, 2)
        # encoded_poses = encoded_poses.to('cpu')
        # reconstructed = encoded_poses.detach().numpy()

        # to tensors
        # word_seq_tensor = words_to_tensor(self.lang_model, word_seq, aux_info['end_time'])
        # pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
            encoded_poses = encoded_poses.reshape((encoded_poses.shape[0], -1)).float()
        # audio = torch.from_numpy(audio).float()


        if self.use_drivitive:
            diff =  [(encoded_poses[n, :] - encoded_poses[n - 1, :]) for n in range(1, encoded_poses.shape[0])]
            diff.insert(0,torch.zeros_like(encoded_poses[0, :]))
            encoded_poses = torch.hstack((encoded_poses, torch.stack(diff)))

        # return word_seq_tensor, pose_seq, audio, aux_info
        return encoded_poses, encoded_poses


    def creat_similarity_dataset(self, pickle_file, labelstxt_file):
        # Todo: 1. Thos function gets the pickle file that I made in the clustering.py(or flowgmm) process as well
        # Todo: as the labels text file that I annotated in the Unity application.
        # Todo: 2. Then I will creat those pairs of similarity and dissimilarity
        # Todo: 3. Finally, we store the pairs into the class.
        # Todo: We will use pairwise label and an extra loss in backpropagation process later.

        # 1. call preprocess load
        self.data_rnn, self.labels, self.pairwise_labels, self.data_original = self.load_gesture_data(pickle_file, labelstxt_file)

        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        self.data_original = ( self.data_original - self.data_mean) / std

        target = torch.from_numpy(self.data_original)

        target = target.float()
        # target = target.to(device)
        with torch.no_grad():
            self.encoded_labeled_poses = self.rep_model.encoder(target)
            if self.use_drivitive:
                diff = [(self.encoded_labeled_poses[n, :] - self.encoded_labeled_poses[n - 1, :])
                        for n in range(1, self.encoded_labeled_poses.shape[0])]
                diff.insert(0, torch.zeros_like(self.encoded_labeled_poses[0, :]))
                self.encoded_labeled_poses = torch.cat((self.encoded_labeled_poses, torch.stack(diff)), dim=2)
        self.pairwise_enabeled = True
        pass


    def get_labeled_(self, count):
        stack_pairs1 = torch.zeros(count, self.encoded_labeled_poses.shape[1], self.encoded_labeled_poses.shape[2])
        stack_pairs2 = torch.zeros(count, self.encoded_labeled_poses.shape[1], self.encoded_labeled_poses.shape[2])
        stack_labels = torch.zeros(count)
        rnds = random.sample(range(1, len(self.pairwise_labels)), 3)
        k=0
        for rnd in rnds:
            current_pair = self.pairwise_labels[rnd]
            s1_ = self.encoded_labeled_poses[current_pair[0]]
            s2_ = self.encoded_labeled_poses[current_pair[1]]
            ss_label = current_pair[2]
            stack_pairs1[k, :, :] = s1_
            stack_pairs2[k, :, :] = s2_
            stack_labels[k] = (ss_label)
            k = k+1


        return stack_pairs1, stack_pairs2, stack_labels

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model

    def load_gesture_data(self, pre_processed_pickle_adress, labelstxt_file):

        loaded = pickle.load(open(pre_processed_pickle_adress, 'rb'))
        liaded_len = len(loaded)
        loaded = np.hstack(loaded)
        print('len loaded', len(loaded))
        print("Loaded successfully")

        data_latent_rnn = []
        data_latent_linear = np.zeros([len(loaded), loaded[0]['latent_linear'].shape[0],
                                       loaded[0]['latent_linear'].shape[1]])
        data_latent_linear_listwise = []
        data_original = []
        count = len(loaded)
        # count = 4000
        for i in range(count):

            # 1
            current_latent_linear = loaded[i]['latent_linear']
            current_latent_rnn = loaded[i]['latent_rnn']
            current_original = loaded[i]['original']

            if len(current_original) != len(loaded[0]['original']):
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

        # Should be constructed here since it is not obvious how many we will have at the end.
        data_latent_linear = np.zeros([len(data_latent_linear_listwise), loaded[0]['latent_linear'].shape[0],
                                       loaded[0]['latent_linear'].shape[1]])
        for i in range(len(data_latent_linear_listwise)):
            data_latent_linear[i] = data_latent_linear_listwise[i]

        data_latent_rnn_ndarray = np.array(data_latent_rnn)

        first_order_labels = np.ones(len(data_latent_rnn_ndarray)) * -1

        labels_list = []
        label_file = open(labelstxt_file, 'r')

        for line in label_file:
            str = line.split(',')
            lbl = (str[4])
            left = int(str[1])
            middle = int(str[2])
            right = int(str[3])
            chance = random.random()
            # if chance > 0.1:
            #     continue
            if (lbl == 'neither'):
                # continue
                labels_list.append([right, middle, 0])
                labels_list.append([left, middle, 0])
                first_order_labels[right] = 1
                first_order_labels[left] = 1
                first_order_labels[middle] = 1
            if (lbl == 'right'):
                labels_list.append([right, middle, 1])
                first_order_labels[right] = 1
                # first_order_labels[left] = 1
                first_order_labels[middle] = 1
            if (lbl == 'left'):
                labels_list.append([left, middle, 1])
                # first_order_labels[right] = 1
                first_order_labels[left] = 1
                first_order_labels[middle] = 1

        #     Todo: read from file

        return data_latent_rnn_ndarray[:, 0:200], first_order_labels, (labels_list), data_original


class TrinityDataset_with_cluster(Dataset):
    def __init__(self, args, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()
        self.use_drivitive = args.use_derivitive == 'True'
        self.kmeanmodel = pickle.load(open('../output/clustering_results/kmeans_model.pk','rb'))

        if args.sentence_level == 'True':
            self.sentence_level = True
        else:
            self.sentence_level = False

        logging.info("Reading data '{}'...".format(lmdb_dir))
        if self.sentence_level:
            preloaded_dir = lmdb_dir + '_sentence_level' + '_cache'
        else:
            preloaded_dir = lmdb_dir + '_cache'

        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # Representation model
        checkpoint_path = args.rep_learning_checkpoint
        self.rep_learning_dim = args.rep_learning_dim
        rep_learning_args, rep_model, rep_loss_fn, rep_lang_model, rep_out_dim = utils.train_utils.load_checkpoint_and_model(
            checkpoint_path, device, 'DAE')
        self.rep_model = rep_model.to('cpu')
        self.rep_model.train(False)
        #   RNN autoencoder
        checkpoint_path = args.autoencoder_checkpoint

        # RNNAutoencoder_args, RNNAutoencoder_model, RNNAutoencoder_fn,\
        # RNNAutoencoder_model, RNNAutoencoder_dim = utils.train_utils.load_checkpoint_and_model(
        #     checkpoint_path, device)

        args, rnn, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
            checkpoint_path, device, 'autoencoder')
        self.RNNAutoencoder_model = rnn.to('cpu')
        self.RNNAutoencoder_model.train(False)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, audio, aux_info, portion = sample

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        pose_seq = (pose_seq - self.data_mean) / std

        # Todo: Here we should apply rep_learning
        target = torch.from_numpy(pose_seq)
        target = target.float()
        with torch.no_grad():
            encoded_poses = self.rep_model.encoder(target)

        encoded_poses = encoded_poses.reshape((encoded_poses.shape[0], -1)).float()

        if self.use_drivitive:
            diff = [(encoded_poses[n, :] - encoded_poses[n - 1, :]) for n in range(1, encoded_poses.shape[0])]
            diff.insert(0, torch.zeros_like(encoded_poses[0, :]))
            encoded_poses = torch.hstack((encoded_poses, torch.stack(diff)))
        with torch.no_grad():
            out_pose, latent, mue, logvar = self.RNNAutoencoder_model(encoded_poses.unsqueeze(0), encoded_poses.unsqueeze(0))
            latent = latent[:self.RNNAutoencoder_model.decoder.n_layers]
            latent = latent.squeeze(1)
            latent = latent.reshape((1, -1)).float()
            # latent = latent.squeeze(0)
            cluster_id = self.kmeanmodel.predict(latent.cpu().detach().numpy())
            cluster_id = torch.Tensor(cluster_id).long()
        # to tensors
        word_seq_tensor = words_to_tensor(self.lang_model, word_seq, aux_info['end_time'])
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()

        # audio = torch.from_numpy(audio).float()

        # return cluster_id
        return word_seq_tensor, encoded_poses, audio, aux_info, cluster_id

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model


class TrinityDataset_sentencelevel(Dataset):
    def __init__(self, args, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.subdivision_stride_sentence = args.subdivision_stride_sentence
        self.args = args
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()
        self.use_drivitive = args.use_derivitive == 'True'

        # Todo: fix adress -- Fixed
        test = os.path.dirname(args.autoencoder_checkpoint)
        self.kmeanmodel = pickle.load(open(os.path.dirname(args.autoencoder_checkpoint)
                                           + '/clusters/kmeans_model.pk', 'rb'))
        # self.clustering_transforms = pickle.load(open(os.path.dirname(args.autoencoder_checkpoint)
        #                                               + '/clusters/transforms.pkl', 'rb'))

        if args.sentence_level == 'True':
            self.sentence_level = True
        else:
            self.sentence_level = False

        logging.info("Reading data '{}'...".format(lmdb_dir))

        if self.sentence_level:
            # preloaded_dir = lmdb_dir + '_sentence_level' + '_cache'
            preloaded_dir = args.model_save_path + 'lmdb/' + os.path.basename(
                (lmdb_dir)) + '_sentence_level' + '_cache'
        else:
            # preloaded_dir = lmdb_dir + '_cache'
            preloaded_dir = args.model_save_path + 'lmdb/' + os.path.basename(
                (lmdb_dir)) + '_cache'


        if not os.path.exists(args.model_save_path + 'lmdb'):
            os.mkdir(args.model_save_path + 'lmdb')
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(args, lmdb_dir, preloaded_dir, n_poses,
                                            self.subdivision_stride_sentence, pose_resampling_fps,
                                            sentence_level=self.sentence_level)
            data_sampler.run()
        else:
            # Todo: Remove later

            # data_sampler = DataPreprocessor(args, lmdb_dir, preloaded_dir, n_poses,
            #                                 subdivision_stride, pose_resampling_fps,
            #                                 sentence_level=self.sentence_level)
            # data_sampler.run()
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))



        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # Representation model
        checkpoint_path = args.rep_learning_checkpoint
        self.rep_learning_dim = args.rep_learning_dim
        rep_learning_args, rep_model, rep_loss_fn, rep_lang_model, rep_out_dim = utils.train_utils.load_checkpoint_and_model(
            checkpoint_path, device, 'DAE')
        self.rep_model = rep_model.to('cpu')
        self.rep_model.train(False)
        #   RNN autoencoder
        checkpoint_path = args.autoencoder_checkpoint

        # RNNAutoencoder_args, RNNAutoencoder_model, RNNAutoencoder_fn,\
        # RNNAutoencoder_model, RNNAutoencoder_dim = utils.train_utils.load_checkpoint_and_model(
        #     checkpoint_path, device)

        args_rnn, rnn, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
            checkpoint_path, device, 'autoencoder_vq')
        self.RNNAutoencoder_model = rnn.to('cpu')
        self.RNNAutoencoder_model.train(False)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, audio_raws, audio_mels, aux_info, sentence_leve_latents, GP3_Embedding = sample

        def words_to_tensor(lang, words, end_time=None):
            # indexes = [lang.SOS_token]
            indexes = []
            if len(words)<=0:
                print()
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            # indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        def poseIndex_to_tensor(sop, eop, poses_indicies):
            indexes = [sop]
            for pose_index in poses_indicies:
                indexes.append(pose_index)
            indexes.append(eop)
            return torch.Tensor(indexes).long()


        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        pose_seq = (pose_seq - self.data_mean) / std


        # to tensors
        word_seq_tensor = words_to_tensor(self.lang_model, word_seq, aux_info['end_time'])
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()

        sentence_leve_latents = torch.from_numpy(sentence_leve_latents)
        sentence_leve_latents = sentence_leve_latents.reshape([sentence_leve_latents.shape[0], -1]).float()

        # Todo: we may need do some preprocessing. (e.g. padding, resampling, etc.)
        # Todo: Move list-->ndarray to the preprocessing.
        audio_raw_for_now = False
        if audio_raw_for_now:
            audio = audio_raws
        else:
            audio = audio_mels
        audio = np.array(audio)
        audio = torch.from_numpy(audio).float()
        # audio = torch.reshape(audio, (sentence_leve_latents.shape[0], -1))
        # audio = torch.reshape(audio, (4, -1))

        # return cluster_id
        cluster_ids = np.zeros(sentence_leve_latents.shape[0])
        if self.RNNAutoencoder_model.vq == True:
            loss_vq, quantized, perplexity_vq, encodings =\
                self.RNNAutoencoder_model.vq_layer(sentence_leve_latents)
            cluster_ids = torch.argmax(encodings, dim=1)

            # q = poseIndex_to_tensor(sop=512, eop=513,
            #                         poses_indicies=cluster_ids.cpu().detach().numpy())
            # cluster_ids = q
            # print(cluster_ids)
        else:
            cluster_ids = self.kmeanmodel.predict(
                # self.clustering_transforms['scalar'].transform
                (sentence_leve_latents.cpu().detach().numpy()))
            cluster_ids = torch.from_numpy(cluster_ids).long()
        # cluster_id = torch.Tensor(cluster_id).long()
        # print(cluster_ids)

        # clusters_portion = self.clustering_transforms['portion'][cluster_ids]
        # clusters_portion = torch.from_numpy(clusters_portion)

        # print((cluster_ids))
        try:
            GP3_Embedding = torch.from_numpy(GP3_Embedding).float()
        except:
            pass

        return word_seq_tensor, pose_seq, audio, aux_info, sentence_leve_latents, cluster_ids, GP3_Embedding

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model