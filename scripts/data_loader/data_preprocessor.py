"""create data samples
"""

import math
import pickle
import os
from typing import Tuple

import lmdb
import numpy as np
import pyarrow
import torch
import librosa
from tqdm import tqdm
from configargparse import argparse
from model.vocab import Vocab

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataPreprocessor:
    """Loads and extracts skeleton, audio and video data from Lmdb files and writes those entires into a separate new Lmdb file.

    Attributes:
        src_lmdb_env: A Lmdb object containing the origin database environment (similar to PostgreSQL schema).
        dst_lmdb_env: A Lmdb object containing the destination database environment.
        n_videos: An integer number of entries in the database (equal to the number of videos in the training set).
        n_poses: An integer number of frames in each clip in the dataset (normally 30 (in 30 fps)).
        subdivision_stride: An integer number of frames between the start of one clip and the start of the next clip (clips can overlap).
        skeleton_resampling_fps: An integer frames per second of clip to use for training (usually downsampled to 20 fps, clips are normally 30 fps).
        audio_sample_length: An integer length of the audio clip in hertz (sampled at 16,000 Hz).
        n_out_samples: An integer total number of database entries (audio, video and skeleton) that has been extracted from the original videos.
        sentence_frame_length: An integer number of frames in each clip but for sentences rather than gestures.
        audio_sampling_rate: An integer sampling rate for an audio signal.
        DAE_frame_level: A DAE model only if args.name in the initialization method is not 'DAE'.
        rnn_representation: A VQVAE model only if 'sentence_level' is True else None.
        ckpt_path_DAE: A string filepath to a saved 'DAE' checkpoint model.
        ckpt_path_Autoencode: A string filepath to a saved VQVAE checkpoint model.
    """
    def __init__(self, args: argparse.Namespace, clip_lmdb_dir: str, out_lmdb_dir: str, n_poses: int, subdivision_stride: int, pose_resampling_fps: int, sentence_level: bool = False):
        """Initialize with several dataset parameters.

        Initializes database connections to the lmdb files. Note that the connections are open until closed by the run method.

        The args argument must contain the following keys:
            name: A string name of the model (ex. 'DAE' or 'autoencoder_vq').
            rep_learning_checkpoint: If name is not 'DAE', a string filepath to a saved 'DAE' checkpoint model.
            autoencoder_checkpoint: If sentence level is True, a string filepath to a saved VQVAE checkpoint model.
            sentence_frame_length: An integer number of frames in each clip (for a sentence instead of gesture).

        Args:
            args: A configargparser object with specific parameters (See above).
            clip_lmdb_dir: A string filepath containing the lmdb dataset files.
            out_lmdb_dir: A string filepath to save output as lmdb files.
            n_poses: An integer number of frames per second in each clip in the dataset (ex. 30 in 30 fps).
            subdivision_stride: An integer number of frames between the start of one clip and the start of the next clip (may overlap).
            pose_resampling_fps: An integer frames per second for training (may differ from clip fps).
            sentence_level: A boolean flag to add a language model.
        """
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.sentence_level = sentence_level
        self.src_lmdb_env: lmdb.Environment = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        self.out_lmdb_dir = out_lmdb_dir
        with self.src_lmdb_env.begin() as txn:
            self.n_videos: int = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        self.ckpt_path_DAE: str = args.rep_learning_checkpoint
        self.ckpt_path_Autoencode: str = args.autoencoder_checkpoint

        if args.name != "DAE":
            self.DAE_frame_level: Tuple[argparse.Namespace, torch.nn.Module, torch.nn.MSELoss, Vocab, int] = utils.train_utils.load_checkpoint_and_model(
                self.ckpt_path_DAE, device,'DAE')

        if self.sentence_level:
            self.rnn_representation: Tuple[argparse.Namespace, torch.nn.Module, torch.nn.MSELoss, Vocab, int] = utils.train_utils.load_checkpoint_and_model(
                self.ckpt_path_Autoencode, device, 'autoencoder_vq')

        # create db for samples
        map_size = 1024 * 50  # in MB
        map_size <<= 20  # in B
        self.dst_lmdb_env: lmdb.Environment = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

        self.sentence_frame_length = args.sentence_frame_length
        self.audio_sampling_rate = 16000


    def run(self) -> None:
        """Extract skeleton, audio, word data from source and write entries into a destination Lmdb file.

        Closes both src_lmdb_env and dst_lmdb_env database connections upon completion.
        Does not return any values. Modifies internal state of the object (Close db connection).
        """
        src_txn = self.src_lmdb_env.begin(write=False)
        total_count = src_txn.stat()['entries']

        # sampling and normalization
        cursor = src_txn.cursor()
        counter = 0
        for key, value in tqdm(cursor):
            print("video ", counter, "of", total_count, '\n')
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip)
                counter = counter + 1

        # print number of samples
        with self.dst_lmdb_env.begin() as txn:
            print("Sample_counter", txn.stat()['entries'])

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip(self, vid: str, clip: dict) -> None:
        """Internal function to extract and write skeleton, audio and word data from provided clip.

        Modifies internal state of the object (n_out_samples, n_poses, audio_sample_length).
        #TODO

        Args:
            vid: A string representing the name or id of the clip.
            clip: A dictionary containing the following string keys:
                'poses': A Numpy array of pose/gesture data.
                'audio_raw': A Numpy array of audio data.
                'words': A list of lists. Each internal list contains 3 elements:
                    index 0: A float start time.
                    index 1: A float end time.
                    index 2: A string word.
        """
        clip_skeleton: np.ndarray = clip['poses']
        clip_audio_raw: np.ndarray = clip['audio_raw']
        clip_word_list: list[list] = clip['words']

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_words_list = []

        sample_audio_list_mels = []
        sample_audio_list_raws = []

        sentence_leve_latents_list = []
        GPT_3_STR_list = []
        GPT_3_Embedding_list = []

        if self.sentence_level:
            self.n_poses = self.sentence_frame_length
            self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * self.audio_sampling_rate)

        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        # Sentence level preparation:

        '''
        self.sentence_level = False
        if self.sentence_level:
            print("Sentence level")
            
            str_transcript = ""
            for i_t in range(len(clip_word_list)):
                str_transcript = str_transcript + " " + clip_word_list[i_t][0]

            import nltk  # library
            # nltk.download()
            sentences = nltk.sent_tokenize(str_transcript)  # whole paragraph break into sentence.

            start_sentence_word_index = 0
            end_sentence_word_index = 0
            for sentence in sentences:
                x = sentence.replace('.', '').strip().split(' ')

                end_sentence_word_index = start_sentence_word_index + len(sentence.replace('.', '').strip().split(' ')) - 1  # to index
                #     process

                reconstructed_sentence = clip_word_list[start_sentence_word_index:end_sentence_word_index+1]

                start_idx = int( clip_word_list[start_sentence_word_index][1] * self.skeleton_resampling_fps )
                try:
                    fin_idx = int( clip_word_list[end_sentence_word_index][2] * self.skeleton_resampling_fps )
                except:

                    print()
                # ........................
                sample_skeletons = clip_skeleton[start_idx:fin_idx]
                subdivision_start_time = start_idx / self.skeleton_resampling_fps
                subdivision_end_time = fin_idx / self.skeleton_resampling_fps
                sample_words = self.get_words_in_time_range(word_list=clip_word_list,
                                                            start_time=subdivision_start_time,
                                                            end_time=subdivision_end_time)
                if len(sample_words) < 2:
                    continue

                # raw audio
                audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
                audio_end = audio_start + self.audio_sample_length
                sample_audio = clip_audio_raw[audio_start:audio_end]

                motion_info = {'vid': vid,
                               'start_frame_no': start_idx,
                               'end_frame_no': fin_idx,
                               'start_time': subdivision_start_time,
                               'end_time': subdivision_end_time}

                sample_skeletons_list.append(sample_skeletons)
                sample_words_list.append(sample_words)
                sample_audio_list.append(sample_audio)
                aux_info.append(motion_info)



                start_sentence_word_index = end_sentence_word_index + 1
                print()
            '''
        # Loading cached GP3 requestes
        address = self.out_lmdb_dir + '_' + vid + '.gpt'
        if os.path.exists(self.out_lmdb_dir + '_' + vid + '.gpt'):
            loaded_gpt3 = pickle.load(open(address, 'rb'))
        else:
            loaded_gpt3 = None

        for i in tqdm(range(num_subdivision)):



            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps
            sample_words = self.get_words_in_time_range(word_list=clip_word_list,
                                                        start_time=subdivision_start_time,
                                                        end_time=subdivision_end_time)


            if len(sample_words) < 4:
                continue

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[audio_start:audio_end]

            mel_chunks = []
            raw_chunks = []
            for audio_sub in range(self.audio_sample_length//self.audio_sampling_rate):
                audio_chunk = sample_audio[audio_sub*self.audio_sampling_rate: (audio_sub+1)*self.audio_sampling_rate]
                signal = librosa.feature.melspectrogram(y=audio_chunk, sr=self.audio_sampling_rate)
                signal = librosa.power_to_db(signal, ref=np.max)
                mel_chunks.append(signal)
                # raw_chunks.append(audio_chunk)
                raw_chunks.append(0)
                # signal = librosa.amplitude_to_db(signal)







            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            sample_words_list.append(sample_words)

            sample_audio_list_mels.append(mel_chunks)
            sample_audio_list_raws.append(raw_chunks)

            aux_info.append(motion_info)
            if self.sentence_level:

                # For the input side:
                # GPT-3 Embeddomg
                str_input = ""
                for word in sample_words:
                    str_input += ' ' + word[0]

                GPT_3_features = self.GPT_3_caller(str_input, loaded_gpt3)
                GPT_3_Embedding_list.append(GPT_3_features)
                GPT_3_STR_list.append(str_input)
                # Discretization for the output side:
                sentence_leve_latents_list.append(self.get_pose_latent(sample_skeletons))

            # if i>100:
            #     break



        object_to_save = {'sample_words_list':GPT_3_STR_list,
                          'GPT_3_Embedding_list': GPT_3_Embedding_list}
        pickle.dump(object_to_save, open(self.out_lmdb_dir + '_' + vid + '.gpt','wb'))
        print("Embedding Saved!")
        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                if self.sentence_level:
                    for words, poses, audio_raws, audio_mels, aux, sentence_leve_latents , GPT_3_Embedding in \
                            zip(sample_words_list, sample_skeletons_list,
                                sample_audio_list_raws, sample_audio_list_mels,
                                aux_info, sentence_leve_latents_list, GPT_3_Embedding_list):
                        poses = np.asarray(poses)
                        GPT_3_Embedding = np.array(GPT_3_Embedding)
                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        v = [words, poses, audio_raws, audio_mels, aux, sentence_leve_latents, GPT_3_Embedding]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
                else:
                    for words, poses, audio, aux in zip(sample_words_list, sample_skeletons_list,
                                                        sample_audio_list, aux_info):
                        poses = np.asarray(poses)

                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        v = [words, poses, audio, aux]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1

    @staticmethod
    def get_words_in_time_range(word_list: list[list], start_time: float, end_time: float) -> list[list]:
        """Retrieves words in the list that fall between the start_time and end_time provided.

        Args:
            word_list: A list that each element contains a list with three elements:
                index 0: A float start time.
                index 1: A float end time.
                index 2: A string word.
            start_time: A float indicating when to start filtering.
            end_time: A float indicating when to end filtering.

        Returns:
            A list containing all elements in the word_list that fall between the start_time and end_time provided.
        """
        words = []

        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]

            if word_s >= end_time:
                break

            if word_e <= start_time:
                continue

            words.append(word)

        return words


    def get_pose_latent(self, poses: np.ndarray) -> np.ndarray:
        """TODO
        """

        # 1. Load models
        args, DAE, loss_fn, lang_model, out_dim = self.DAE_frame_level
        args, rnn, loss_fn, lang_model, out_dim = self.rnn_representation
        DAE.train(False)
        rnn.train(False)


        # poses, poses_mirror = process_bvh(bvh_file)

        mean = np.array(args.data_mean).squeeze()
        std = np.array(args.data_std).squeeze()
        std = np.clip(std, a_min=0.01, a_max=None)
        out_poses = (np.copy(poses) - mean) / std

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
        # all_frames_from_rnn = None

        result = np.zeros((len(encoded)//args.n_poses, rnn.decoder.n_layers, args.hidden_size))
        result_index = 0
        for i in range(0, len(encoded), args.n_poses):
            current_dict = dict()
            input_seq = encoded[i:i + args.n_poses]

            current_dict['original'] = poses[i: i + args.n_poses]
            current_dict['latent_linear'] = encoded[i: i + args.n_poses].detach().clone().to('cpu').numpy()

            input_pre_seq = encoded[i]
            output_seq = encoded[i:i + args.n_poses]

            input_seq = torch.unsqueeze(input_seq, 0)
            input_seq = input_seq.transpose(0, 1)
            use_drivitive = args.use_derivitive=='True'
            if use_drivitive:
                diff = [(input_seq[n, :] - input_seq[n - 1, :]) for n in range(1, input_seq.shape[0])]
                diff.insert(0, torch.zeros_like(input_seq[0, :]))
                input_seq = torch.cat((input_seq, torch.stack(diff)), dim=2)

            # output_seq = torch.unsqueeze(output_seq, 0)
            # output_seq = output_seq.transpose(0, 1)
            output_seq = input_seq.clone()




            reconstructed_rnn = torch.zeros(args.n_poses, output_seq.size(1), rnn.decoder.output_size) \
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
                # print("decoder_hidden!!! modified", decoder_hidden.shape)
                decoder_first_hidden = decoder_hidden
            else:
                decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder
                # print("decoder_hidden!!! not VAE ", decoder_hidden.shape)

            current_dict['latent_rnn'] = torch.squeeze(decoder_hidden.detach().clone(), 1).to('cpu').numpy()
            result[result_index] = torch.squeeze(decoder_hidden.detach().clone(), 1).to('cpu').numpy()
            result_index = result_index + 1

        return result

    def GPT_3_caller(self, str_input, pickle_file_loaded):

        return 1

        if pickle_file_loaded != None:
            for i in range(len(pickle_file_loaded['sample_words_list'])):
                if str_input == pickle_file_loaded['sample_words_list'][i]:
                    print("Embedding found:", str_input)
                    return pickle_file_loaded['GPT_3_Embedding_list'][i]

        request = openai.Embedding.create(input=str_input, engine="text-similarity-ada-001")
        embedding = request['data'][0]['embedding']
        print("!!!!!!!!!!!!!!!!!!!Embedding requested:", str_input)
        return embedding

