"""
"""


import argparse
import os
import glob
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import lmdb
import pyarrow
from sklearn.pipeline import Pipeline
from scipy.spatial.transform import Rotation as R
import joblib as jl

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from utils.data_utils import SubtitleWrapper, normalize_string

# 15 standard joints
target_joints = [
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Neck1",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
]
rep_learning = False


def process_bvh(gesture_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a bvh file to numpy arrays.

    Several parameters are fixed in this function as follows:
        - 20 frames per second for each clip.
        - 'hip_centric' model.
        - To create more datapoints, mirror on the 'x' axis.
        - 15 joints included in 'target_joints' global variable above.

    Args:
        gesture_filename: The string filename of a specific gesture (bvh) file.

    Returns:
        A 2-Tuple:
            out_matrix[0]: Numpy array of the gestures.
            out_matrix[1]: Numpy array of the mirrored gestures.
    """
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline(
        [
            ("dwnsampl", DownSampler(tgt_fps=20, keep_all=False)),
            ("root", RootTransformer("hip_centric")),
            ("mir", Mirror(axis="X", append=True)),
            ("jtsel", JointSelector(target_joints, include_root=True)),
            ("cnst", ConstantsRemover()),
            ('np', Numpyfier())
        ]
    )

    out_data: np.ndarray = data_pipe.fit_transform(data_all)
    jl.dump(data_pipe, os.path.join("../resource", "data_pipe.sav"))

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 3))
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 9))
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            r = R.from_euler("ZXY", out_data[i, j], degrees=True)
            out_matrix[i, j] = r.as_matrix().reshape(out_data.shape[2], 9)
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0], out_matrix[1]


def make_lmdb_gesture_dataset(args: argparse.Namespace) -> None:
    """Converts the dataset (audio, bvh and transcripts) into LMDB files.

    Main function. Saves the files into LMDB files in a new folder ('lmdb') within a directory.
    The data is separated into train and test datasets.

    Args:
        args: A configargparse object containing several parameters
            - 'db_path': The string file path that contains the files.
    """
    base_path: str = args.db_path
    gesture_path = os.path.join(base_path, "Motion")
    audio_path = os.path.join(base_path, "Audio")
    text_path = os.path.join(base_path, "Transcripts")
    if not rep_learning:
        out_path = os.path.join(base_path, "lmdb")
    else:
        out_path = os.path.join(base_path, "lmdb_rep_learning")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 20  # in MB
    map_size <<= 20  # in B
    db: list[lmdb.Environment] = [
        lmdb.open(os.path.join(out_path, "lmdb_train"), map_size=map_size),
        lmdb.open(os.path.join(out_path, "lmdb_test"), map_size=map_size),
    ]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_poses = []
    bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
    for v_i, bvh_file in enumerate(bvh_files):
        name = os.path.split(bvh_file)[1][:-4]
        print(name)

        # load skeletons and subtitles
        poses, poses_mirror = process_bvh(bvh_file)
        subtitle = SubtitleWrapper(os.path.join(text_path, name + ".json")).get()

        # load audio
        audio_raw, audio_sr = librosa.load(
            os.path.join(audio_path, "{}.wav".format(name)),
            mono=True,
            sr=16000,
            res_type="kaiser_fast",
        )

        # process
        clips = [
            {"vid": name, "clips": []},  # train
            {"vid": name, "clips": []},  # validation
        ]

        # split
        if v_i == 0:
            dataset_idx = 1  # validation
        else:
            dataset_idx = 0  # train

        # word preprocessing
        word_list = []
        for wi in range(len(subtitle)):
            word_s = float(subtitle[wi]["start_time"][:-1])
            word_e = float(subtitle[wi]["end_time"][:-1])
            word = subtitle[wi]["word"]

            word = normalize_string(word)
            if len(word) > 0:
                word_list.append([word, word_s, word_e])

        # save subtitles and skeletons
        poses = np.asarray(poses, dtype=np.float16)
        clips[dataset_idx]["clips"].append(
            {"words": word_list, "poses": poses, "audio_raw": audio_raw}
        )
        poses_mirror = np.asarray(poses_mirror, dtype=np.float16)
        clips[dataset_idx]["clips"].append(
            {"words": word_list, "poses": poses_mirror, "audio_raw": audio_raw}
        )

        # write to db
        for i in range(2):
            with db[i].begin(write=True) as txn:
                if len(clips[i]["clips"]) > 0:
                    k = "{:010}".format(v_i).encode("ascii")
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)

        all_poses.append(poses)

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()

    # calculate data mean
    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0)
    pose_std = np.std(all_poses, axis=0)

    print("data mean/std")
    print(str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print(str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    args = parser.parse_args()
    make_lmdb_gesture_dataset(args)
