import argparse
import glob
import os
from pathlib import Path

import librosa
import lmdb
import pyarrow
from sklearn.pipeline import Pipeline
import joblib as jl

from pymo.preprocessing import *
from pymo.parsers import BVHParser
from utils.data_utils import SubtitleWrapper, normalize_string

# 18 joints (only upper body)
target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 24 joints (upper and lower body excluding fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 56 joints (upper and lower body including fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']


target_joints_trinity = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
                 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

bvh_file_triity = '/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh'
bvh_file_twh = '/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/GENEA/Co-Speech_Gesture_Generation/dataset/dataset_v1/val/bvh/val_2022_v1_030.bvh'


def process_bvh_trinity(gesture_filename):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=20, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('mir', Mirror(axis='X', append=True)),
        ('jtsel', JointSelector(target_joints_trinity, include_root=True)),
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())
    ])

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=20, keep_all=False)),
        ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        # ('mir', Mirror(axis='X', append=True)),
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    jl.dump(data_pipe, os.path.join('../resource', 'data_pipe.sav'))

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 3))
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 9))
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            r = R.from_euler('ZXY', out_data[i, j], degrees=True)
            out_matrix[i, j] = r.as_matrix().reshape(out_data.shape[2], 9)
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0], out_matrix[1]

def process_bvh_twh(gesture_filename, dump_pipeline=False):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=20, keep_all=False)),
        ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        # ('mir', Mirror(axis='X', append=True)),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    if dump_pipeline:
        jl.dump(data_pipe, os.path.join('../resource', 'data_pipe.sav'))

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6))  # 3 pos (XYZ), 3 rot (ZXY)
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 12))  # 3 pos, 1 rot matrix (9 elements)
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            for k in range(out_data.shape[2]):  # joints
                out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0]

def process_bvh_rot_only(gesture_filename, dump_pipeline=False):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=20, keep_all=False)),
        # ('root', RootTransformer('hip_centric')),
        ('root', RootNormalizer()),
        # ('mir', Mirror(axis='X', append=True)),
        ('jtsel', JointSelector(target_joints, include_root=True)),
        # ('cnst', ConstantsRemover()),
        # ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    jl.dump(data_pipe, os.path.join('../resource', 'data_pipe.sav'))

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 3))
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 9))
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            r = R.from_euler('ZXY', out_data[i, j], degrees=True)
            out_matrix[i, j] = r.as_matrix().reshape(out_data.shape[2], 9)
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0], out_matrix[1]


# trinity_a = process_bvh_trinity(bvh_file_triity)
trinity_a = process_bvh_trinity(bvh_file_twh)

twh_a = process_bvh_twh(bvh_file_twh)
print("Test")