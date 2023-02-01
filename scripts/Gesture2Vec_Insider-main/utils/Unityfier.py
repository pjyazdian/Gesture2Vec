# This file converts transcript to a txt file that can be read by the unity
import json
import glob
import os
from data_utils import SubtitleWrapper, normalize_string

jsons_path = '/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts'

json_output_path = jsons_path + '/Unity'
if not os.path.exists(json_output_path):
    os.makedirs(json_output_path)



json_files = sorted(glob.glob(jsons_path + "/*.json"))


for jfile in json_files:
    name = os.path.split(jfile)[1][:-5]
    print(name)

    subtitle = SubtitleWrapper(jfile).get()
    str_subtitle = ''
    for word_boundle in subtitle:
        start_time = word_boundle['start_time'][:-1] #Removing 's'
        end_time = word_boundle['end_time'][:-1]     #Removing 's'
        word = word_boundle['word']
        str_subtitle += "{},{},{}\n".format(start_time, end_time, word)

    str_subtitle = str_subtitle[:-1]

    file2write = open(json_output_path + '/' + name + '.txt', 'w')
    file2write.write(str_subtitle)
    file2write.flush()
    file2write.close()

    print()