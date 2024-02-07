import argparse
import math
import pickle
import pprint
import random
from pathlib import Path
from datetime import datetime

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
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger

from data_loader.data_preprocessor import DataPreprocessor

from inference_Autoencoder import smoothing_function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def mk_hs_slots(tags_list, name, transcript_path, dict_, result):
    print(name)
    transcript = SubtitleWrapper(transcript_path).get()

    word_list = []
    for wi in range(len(transcript)):
        word_s = float(transcript[wi]['start_time'][:-1])
        word_e = float(transcript[wi]['end_time'][:-1])
        word = transcript[wi]['word']

        word = normalize_string(word)
        if len(word) > 0:
            word_list.append([word, word_s, word_e])

    q = 0
    for word in word_list:
        if word[0] == 'like':
            q = q + 1

    for tag in tags_list:
        for word in word_list:
            if word[0] == str.lower(tag):
                dict_[tag] += 1
                result.append((name, word))




    # print("Wait")
    return dict_, result

if __name__ == '__main__':

    '''
        ../output/text2embedding/text2embedding_checkpoint_100.bin
        /local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/TestSeq001.json
        ../output/DAE/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
        ../output/autoencoder/VAE+sim/Dautoencoder_fxw_zinput_VAE_checkpoint_020.bin


       ../output/text2embedding/text2embedding_300d/text2embedding_300d_checkpoint_160.bin
/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/TestSeq001.json
../output/DAE/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
../output/autoencoder/VAE+sim/Decoder_trained/autoencoder_decoder_trained_checkpoint_020.bin


              ../output/text2embedding_ABST/text2embedding_ABST_checkpoint_100.bin
/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/TestSeq001.json
../output/DAE/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
../output/autoencoder/autoencoder_both_traned_sametime/autoencoder_both_traned_sametime_checkpoint_100.bin

       '''
    random.seed(datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("transcript_path", type=Path)
    parser.add_argument("rep_learning_checkpoint", type=Path)
    parser.add_argument("autoencoder_checkpoint", type=Path)

    args = parser.parse_args()

    pre_transcript_path = "/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/"


    # Read softbank tags
    file = open("../output/hs_testcases/softbank_tags.txt", 'r')
    tags_list = []
    for line in file:
        seprated = line.split(',')
        tags_list.append(str.lower(seprated[0]))

    dict_ = dict()
    result = []
    for tag in tags_list:
        dict_[tag] = 0
    for i in reversed(range(1, 11)):
        transcript_path = pre_transcript_path + "TestSeq" + str(i).zfill(3) + ".json"
        transcript_path = Path(transcript_path)

        dict_, result  = mk_hs_slots(tags_list, "TestSeq" + str(i).zfill(3), transcript_path, dict_, result)
    total_categiry = 0
    total = 0
    for key in dict_:
        if dict_[key]>0:
            total_categiry+=1
            total += dict_[key]
            print(key, ",", dict_[key])

    file = open("../output/hs_testcases/word2slot.txt", 'w')
    for item in result:
        str1 = item[0] + "," + item[1][0] + "," + str(item[1][1]) + "," + str(item[1][2])
        file.write(str1 + '\n')
    file.flush()
    file.close()
    print("done!", total, "of", dict_.__len__())