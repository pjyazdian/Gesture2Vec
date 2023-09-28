"""This file converts transcript to a txt file that can be read by the unity.

This script is meant to be run separately.
This file expects JSON files. jsons_path variable must be manually changed to the appropriate directory.
The expected structure of the original JSON file is as follow:
    Subtitles are expected to be contained in a JSON file with the format:
    {
        'alternative':
            []: # only the first element contains the following:
                {
                    'words': [
                        {
                            'start_time': '0.100s',
                            'end_time': '0.500s',
                            'word': 'really'
                        },
                        { <contains more of these structured elements> }
                    ],
                    <other data>
                }
    }
    Note: JSON uses double-quotes instead of single-quotes. Single quotes are used for doc-string reasons.

The files are saved in a new folder in the jsons_path directory named 'Unity'.
"""


import glob
import os
from data_utils import SubtitleWrapper

jsons_path = "/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts"

json_output_path = jsons_path + "/Unity"
if not os.path.exists(json_output_path):
    os.makedirs(json_output_path)


json_files = sorted(glob.glob(jsons_path + "/*.json"))


for jfile in json_files:
    name = os.path.split(jfile)[1][:-5]
    print(name)

    subtitle = SubtitleWrapper(jfile).get()
    str_subtitle = ""
    for word_boundle in subtitle:
        start_time = word_boundle["start_time"][:-1]  # Removing 's'
        end_time = word_boundle["end_time"][:-1]  # Removing 's'
        word = word_boundle["word"]
        str_subtitle += "{},{},{}\n".format(start_time, end_time, word)

    str_subtitle = str_subtitle[:-1]

    file2write = open(json_output_path + "/" + name + ".txt", "w")
    file2write.write(str_subtitle)
    file2write.flush()
    file2write.close()

    print()
