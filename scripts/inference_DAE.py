"""This scripts run inference for Part a: Pose Representation Learning.

The flag variable DATASET_Type should be set to either 'trinity' or 'twh'.
Trinity expects 135 dimensions for gesture data.
Twh expects 160 dimensions for gesture data.

Typical usage example:
    python inference_DAE.py <part a checkpoint path>

Note: checkpoint paths should specify the file (ex. ../output/DAE/model_checkpoint_100.bin).
"""


from __future__ import annotations
import argparse
import os
import pprint
from pathlib import Path

import argparse
from typing import Tuple
import numpy as np
from scipy.signal import savgol_filter
import joblib as jl
from openTSNE import TSNE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

import torch
from torchvision.utils import save_image

import utils
from pymo.preprocessing import *
from pymo.preprocessing import MocapParameterizer
from pymo.viz_tools import *
from pymo.writers import *
from tqdm import tqdm

from utils.train_utils import load_checkpoint_and_model
from trinity_data_to_lmdb import process_bvh as process_bvh_trinity
from twh_dataset_to_lmdb import process_bvh_rot_only_Taras as process_bvh_rot_only_Taras
from twh_dataset_to_lmdb import process_bvh_test1 as process_bvh_rot_test1
from model.DAE_model import DAE_Network, VQ_Frame

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_Type = "Trinity"  # or 'TWH'


def generate_gestures(
    args: argparse.Namespace, pose_decoder: torch.nn.Module, bvh_file: str
) -> (
    Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]
    | Tuple[np.ndarray, np.ndarray, torch.Tensor]
):
    """Build gestures starting from input bvh file.

    The 'args' argument must contain the following keys:
        data_mean: A list of float means from each clip in the dataset.
        data_std: A list of float std from each clip in the dataset.
        autoencoder_vq: A string boolean if VQVAE was trained.

    Args:
        args: A configargparser object with specific keys (See above).
        pose_decoder: A pretrained Part a net.
        bvh_file: A string filepath to a bvh file.

    Returns:
        A 4-Tuple or 3-Tuple:
            Case 1 - 4-Tuple (if args.autoencoder_vq is 'True'):
                out_poses: An array of (standardized) actual gesture data.
                reconstructed: An array of predicted gesture data.
                latent: A Tensor of the VQVAE loss.
                encodings: A Tensor of the VQVAE perplexity score.
            Case 2 - 3-Tuple:
                out_poses: An array of (standardized) actual gesture data.
                reconstructed: An array of predicted gesture data.
                latent: A Tensor of latent code space.

    """
    if DATASET_Type == "Trinity":
        poses, poses_mirror = process_bvh_trinity(bvh_file)
    elif DATASET_Type == "TWH":
        poses = process_bvh_rot_test1(bvh_file)
        poses = poses[250:600]
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    out_poses = (poses - mean) / std

    target = torch.from_numpy(out_poses)
    target = torch.unsqueeze(target, 2)
    target = target.to(device).float()
    reconstructed = []
    # for i in range(len(out_poses)):
    #     input = torch.unsqueeze(target[i],0)
    #     current_out = pose_decoder(input)
    #     reconstructed.append(current_out)
    if args.autoencoder_vq == "True":
        check_prototypes(pose_decoder)
        reconstructed, latent, encodings = pose_decoder(target, Inference=True)
    else:
        reconstructed, latent = pose_decoder(target, get_latent=True)
    reconstructed = torch.squeeze(reconstructed, 2)
    reconstructed = reconstructed.to("cpu")
    reconstructed = reconstructed.detach().numpy()

    if latent is not None:  # for H-1 case
        latent = latent.cpu().detach().numpy()

    if args.autoencoder_vq == "True":
        encodings = encodings.detach().cpu().numpy()
        return out_poses, np.array(reconstructed), latent, encodings
    else:
        return out_poses, np.array(reconstructed), latent


def check_prototypes(model: torch.nn.Module) -> None:
    """Plot VQVAE embedding layer weights.

    Args:
        model: A Part a (DAE) pretrained model.
    """
    Embeddings = model.vq_layer._embedding
    weights = Embeddings.weight
    print(weights.shape)
    dists = torch.cdist(weights, weights)
    dists = dists.cpu().detach().numpy()
    plt.imshow(dists)
    plt.show()


def k_components_analysis_KMEANS(latent: np.ndarray) -> None:
    mms = MinMaxScaler()
    mms.fit(latent)
    data_transformed = mms.transform(latent)

    Sum_of_squared_distances = []
    silhouette_scores = []
    K = range(20, 100, 1)
    for k in tqdm(K):
        km = KMeans(n_clusters=k, max_iter=2500, random_state=0)
        km = km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)
        s_score = silhouette_score(data_transformed, km.labels_)
        silhouette_scores.append(s_score)

    plt.plot(K, Sum_of_squared_distances, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum_of_squared_distances")
    plt.title("Elbow Method For Optimal k")
    plt.show()

    plt.plot(K, silhouette_scores, "bx-")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method For Optimal k")
    plt.show()

    exit()


def k_components_analysis_VQ(base_folder: str, mink: int, maxk: int) -> None:
    Sum_of_squared_distances = []
    silhouette_scores = []

    for model_id in range(mink, maxk):
        checkpoint_path = (
            base_folder + "/" + str(model_id) + "/DAE_H40_checkpoint_030.bin"
        )
        # 1. load model
        (
            args,
            generator,
            loss_fn,
            lang_model,
            out_dim,
        ) = utils.train_utils.load_checkpoint_and_model(checkpoint_path, device, "DAE")

        bvh_file = "../../data/Test_data/Motion/TestSeq001.bvh"
        org_poses, reconstructed, latent, encodings = generate_gestures(
            args, generator, bvh_file
        )

        # Generate latents
        indices = np.argmax(encodings, axis=1)
        centers = generator.vq_layer._embedding.weight.cpu().detach().numpy()
        k_component = generator.vq_components

        mms = MinMaxScaler()
        mms.fit(latent)
        data_transformed = mms.transform(latent)

        # Sum_of_squared_distances.append(km.inertia_)
        s_score = silhouette_score(data_transformed, indices)
        silhouette_scores.append(s_score)

        # plt.plot(K, Sum_of_squared_distances, 'bx-')
        # plt.xlabel('k')
        # plt.ylabel('Sum_of_squared_distances')
        # plt.title('Elbow Method For Optimal k')
        # plt.show()

    plt.plot(range(mink, maxk), silhouette_scores, "bx-")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method For Optimal k")
    plt.show()

    exit()


def Save4Unity(
    latents: np.ndarray,
    indecies: list,
    components: int,
    kernels: np.ndarray,
    save_path: str,
) -> None:
    # Save for unity:

    # 0. prep
    f = open(save_path + "/latents.txt", "w")
    f.write(str(components) + "\n")

    # Fitting transformations
    # pca = PCA(n_components=20)
    # priciple_components = pca.fit(latents)

    MyTSNE = TSNE(
        n_components=2,
        perplexity=200,
        metric="euclidean",
        n_jobs=8,
        verbose=True,
        n_iter=5000,
        early_exaggeration_iter=300,
    )
    combined = np.concatenate((kernels, latents), axis=0)
    TSNE_Transfotm = MyTSNE.fit(combined)

    transformed_kernels = TSNE_Transfotm[0 : len(kernels)]
    transformed_latents = TSNE_Transfotm[len(kernels) :]

    # 1. Embeddings:
    for i in range(0, len(transformed_kernels)):
        line = (
            "{:.3f}".format(transformed_kernels[i, 0])
            + ","
            + "{:.3f}".format(transformed_kernels[i, 1])
        )
        f.write(line + "\n")

    # 2. latents
    for i in range(0, len(transformed_latents)):
        line = str(i) + "," "{:.3f}".format(
            transformed_latents[i, 0]
        ) + "," + "{:.3f}".format(transformed_latents[i, 1]) + "," + str(indecies[i])
        f.write(line + "\n")


def plot_latent(
    args: argparse.Namespace,
    latents: np.ndarray,
    model: torch.nn.Module | DAE_Network | VQ_Frame,
) -> None:
    """Plot the latent code space of a Part a (DAE) pretrained model.

    The 'args' argument must have the following keys:
        model_save_path: A string directory to save the plots.

    Args:
        args: A configargparser object with specific keys (See above).
        latents: An array of the latent code space.
        model: A Part a (DAE) pretrained model.
    """
    # TSNE

    MyTSNE = TSNE(
        n_components=2, perplexity=30, metric="euclidean", n_jobs=8, verbose=True
    )
    X_embedded = MyTSNE.fit(latents)

    plt.figure(figsize=(16, 10))
    # 2D
    sns.scatterplot(
        X_embedded[:, 0],
        X_embedded[:, 1],
        legend=False,
        palette=sns.color_palette("Set2"),
    )

    address = (args.model_save_path) + "/"
    os.makedirs(address, exist_ok=True)
    address += "Embedding_latent.png"
    plt.savefig(address)

    plt.show()

    # Heatmap
    x_lim, y_lim = 60, 60

    heatmap, x_edge, y_edge = np.histogram2d(
        X_embedded[:, 0], X_embedded[:, 1], bins=100
    )
    extent = [x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]]
    extent = [-y_lim, y_lim, -x_lim, x_lim]
    plt.clf()
    sigma = 2
    for sigma in range(1, 3):
        plt.figure(figsize=(16, 10))
        heatmap_filtered = gaussian_filter(heatmap, sigma)
        plt.imshow(heatmap_filtered.T, extent=extent, origin="lower")
        plt.colorbar()
        plt.title("Heatmap of representation dataset, Sigma = " + str(sigma))
        plt.show()
        plt.clf()

    w: torch.Tensor = model.encoder[0].weight
    w = w.cpu().detach().numpy()

    scaler = StandardScaler()
    scaler.fit(w)
    # Todo: check which one is better? Unnormalized showed better results.
    normalized_data = scaler.transform(w)

    max = np.max(normalized_data)
    min = np.min(normalized_data)

    plt.imshow(w)
    plt.title("Kernel w ({},{})".format(np.min(w), np.max(w)))
    plt.show()

    if isinstance(model, VQ_Frame):
        if model.vae:
            # Plot Variance kernel
            w = model.VAE_fc_std.weight
            w = w.cpu().detach().numpy()
            plt.imshow(w)
            plt.title("Kernel of STD layer ({},{})".format(np.min(w), np.max(w)))
            plt.show()

            # Plot Variance kernel
            w = model.VAE_fc_mean.weight
            w = w.cpu().detach().numpy()
            plt.imshow(w)
            plt.title("Kernel of Mean layer ({},{})".format(np.min(w), np.max(w)))
            plt.show()

    return


def Plot_Kernel(_model: DAE_Network | VQ_Frame, args: argparse.Namespace) -> None:
    """Plot for visualizing the learned weights of the autoencoder's encoder.

    The 'args' argument must have the following keys:
        data_mean: A list of float means from each clip in the dataset.
        data_std: A list of float std from each clip in the dataset.

    Args:
        _model: A Part a (DAE) pretrained net.
        args: A configargparser object with specific keywords (See above).
    """

    def to_eular(input_pose: np.ndarray, args: argparse.Namespace) -> np.ndarray:
        # unnormalize
        mean = np.array(args.data_mean).squeeze()
        std = np.array(args.data_std).squeeze()
        std = np.clip(std, a_min=0.01, a_max=None)
        input_pose = np.multiply(input_pose, std) + mean

        # rotation matrix to euler angles
        out_poses = input_pose.reshape((-1, 9))
        out_poses = out_poses.reshape((out_poses.shape[0], 3, 3))
        out_euler = np.zeros((out_poses.shape[0] * 3))

        r = R.from_matrix(out_poses)
        out_euler = r.as_euler("ZXY", degrees=True).flatten()

        pipeline: Pipeline = jl.load("../resource/data_pipe.sav")
        out_euler = np.expand_dims(out_euler, axis=0)
        bvh_data = pipeline.inverse_transform([out_euler])
        return bvh_data

    w = _model.encoder[0].weight
    w = w.cpu().detach().numpy()

    plt.imshow(w)
    plt.title("Kernel w ({},{})".format(np.min(w), np.max(w)))
    plt.show()

    for i in range(len(w)):
        bvh_data = to_eular(w[i], args)
        q = MocapParameterizer("position")
        XX = q.transform(bvh_data)
        ax = draw_stickfigure(XX[0], 0, XX[0].values)
        plt.show()

        plt.imshow(w[i].reshape((15, 9)))
        plt.title("15,9 style")
        plt.show()
    pass


def main(checkpoint_path: str):
    """Main inference function for Part a.

    Args:
        checkpoint_path: A string filepath to a checkpoint model and parameters.
    """
    (
        args,
        generator,
        loss_fn,
        lang_model,
        out_dim,
    ) = load_checkpoint_and_model(checkpoint_path, device, "DAE")
    pprint.pprint(vars(args))
    save_path = os.path.dirname(checkpoint_path)
    os.makedirs(save_path, exist_ok=True)

    # load lang_model
    # vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    # with open(vocab_cache_path, 'rb') as f:
    #     lang_model = pickle.load(f)

    # prepare input
    # transcript = SubtitleWrapper(transcript_path).get()

    # inference
    # 1. Trinity Dataset
    if DATASET_Type == "Trinity":
        bvh_file = "../../data/Test_data/Motion/TestSeq001.bvh"
    # 2. Talking With Hands 16.2M
    elif DATASET_Type == "TWH":
        bvh_file = "/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/GENEA/Co-Speech_Gesture_Generation/dataset/dataset_v1/val/bvh/val_2022_v1_012.bvh"

    if args.autoencoder_vq == "True":
        org_poses, reconstructed, latent, encodings = generate_gestures(
            args, generator, bvh_file
        )
    else:
        org_poses, reconstructed, latent = generate_gestures(args, generator, bvh_file)

    # Plot_Kernel(_model=generator, args=args)
    try:
        plot_latent(args, latent, generator)
    except Exception as e:
        print("Exception in plot_latent", e)

    # unnormalize
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    reconstructed = np.multiply(reconstructed, std) + mean
    org_poses = np.multiply(org_poses, std) + mean

    # make a BVH
    filename_prefix = "{}".format("test_original_DAE")
    make_bvh(save_path, filename_prefix, org_poses)
    filename_prefix = "{}".format("test_reconstructed_DAE")
    make_bvh(save_path, filename_prefix, reconstructed)


def make_bvh(save_path: str, filename_prefix: str, poses: np.ndarray) -> None:
    """Save input gesture data into a bvh file.

    This function requires a saved Pipeline file located in:
    '../resource/data_pipe.sav'.

    Args:
        save_path: A string directory to save the
        filename_prefix: A string filename to use for the saved file.
        poses: An array of gestures data.
    """
    if DATASET_Type == "TWH":
        return make_bvh_TWH(save_path, filename_prefix, poses)
    else:
        return make_bvh_Trinity(save_path, filename_prefix, poses)


def make_bvh_Trinity(save_path: str, filename_prefix: str, poses: np.ndarray) -> None:
    """Save Trinity input gesture data into a bvh file.

    Trinity data contains 135 dimensions of gestures.

    This function requires a saved Pipeline file located in:
    '../resource/data_pipe.sav'.

    Args:
        save_path: A string directory to save the
        filename_prefix: A string filename to use for the saved file.
        poses: An array of gestures data.
    """
    writer = BVHWriter()
    pipeline: Pipeline = jl.load("../resource/data_pipe.sav")

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))

    if n_poses > 15 and False:
        for i in range(poses.shape[1]):
            out_poses[:, i] = savgol_filter(
                poses[:, i], 15, 2
            )  # NOTE: smoothing on rotation matrices is not optimal
    else:
        out_poses = poses

    # rotation matrix to euler angles
    out_poses = out_poses.reshape(
        (out_poses.shape[0], -1, 9)
    )  # (n_frames, n_joints, 9)
    out_poses = out_poses.reshape((out_poses.shape[0], out_poses.shape[1], 3, 3))
    out_euler = np.zeros((out_poses.shape[0], out_poses.shape[1] * 3))
    for i in range(out_poses.shape[0]):  # frames
        r = R.from_matrix(out_poses[i])
        out_euler[i] = r.as_euler('ZXY', degrees=True).flatten()

    bvh_data = pipeline.inverse_transform([out_euler])

    out_bvh_path = os.path.join(save_path, filename_prefix + ".bvh")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(out_bvh_path, "w") as f:
        writer.write(bvh_data[0], f)


def make_bvh_TWH(save_path: str, filename_prefix: str, poses: np.ndarray) -> None:
    """Save TWH input gesture data into a bvh file.

    TWH data contains 160 dimensions of gestures.

    This function requires a saved Pipeline file located in:
    '../resource/data_pipe.sav'.

    Args:
        save_path: A string directory to save the
        filename_prefix: A string filename to use for the saved file.
        poses: An array of gestures data.
    """
    writer = BVHWriter()
    pipeline: Pipeline = jl.load("../resource/data_pipe_TWH.sav")

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))

    for i in range(poses.shape[1]):
        out_poses[:, i] = savgol_filter(
            poses[:, i], 15, 2
        )  # NOTE: smoothing on rotation matrices is not optimal

    # rotation matrix to euler angles
    out_poses = out_poses.reshape(
        (out_poses.shape[0], -1, 12)
    )  # (n_frames, n_joints, 12)
    out_data = np.zeros((out_poses.shape[0], out_poses.shape[1], 6))
    for i in range(out_poses.shape[0]):  # frames
        for j in range(out_poses.shape[1]):  # joints
            out_data[i, j, :3] = out_poses[i, j, :3]
            r = R.from_matrix(out_poses[i, j, 3:].reshape(3, 3))
            out_data[i, j, 3:] = r.as_euler("ZXY", degrees=True).flatten()

    out_data = out_data.reshape(out_data.shape[0], -1)
    bvh_data = pipeline.inverse_transform([out_data])

    out_bvh_path = os.path.join(save_path, filename_prefix + "_generated.bvh")
    with open(out_bvh_path, "w") as f:
        writer.write(bvh_data[0], f)


def feat2bvh(save_path: str, filename_prefix: str, poses: np.ndarray) -> None:
    """Save input gesture data into a bvh file.

    This function requires a saved Pipeline file located in:
    '../resource/data_pipe.sav'.

    Args:
        save_path: A string directory to save the
        filename_prefix: A string filename to use for the saved file.
        poses: An array of gestures data.
    """
    writer = BVHWriter()
    pipeline: Pipeline = jl.load("../resource/data_pipe.sav")

    # transform the data back to it's original shape
    # note: in a real scenario this is usually done with predicted data
    # note: some transformations (such as transforming to joint positions) are not inversible
    poses = pipeline.inverse_transform([poses])

    # ensure correct body orientation
    # poses[0].values["body_world_Xrotation"] = 0
    # poses[0].values["body_world_Yrotation"] = 0
    # poses[0].values["body_world_Zrotation"] = 0

    # Test to write some of it to file for visualization in blender or motion builder
    out_bvh_path = os.path.join(save_path, filename_prefix + ".bvh")
    with open(out_bvh_path, "w") as f:
        writer.write(poses[0], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    # k_components_analysis_VQ('../output/DAE_New/VQs', 5, 200)

    main(args.ckpt_path)

    # Double-check: process_bvh & make_bvh pipeline
    # 1. Load
    # bvh_file = "../../data/Test_data/Motion/TestSeq001.bvh"
    # loaded = process_bvh_trinity(bvh_file)
    #
    # # Normalize
    # mean = np.array(args.data_mean).squeeze()
    # std = np.array(args.data_std).squeeze()
    # std = np.clip(std, a_min=0.01, a_max=None)
    # normalized_poses = (np.copy(loaded) - mean) / std
    #
    # # Un-Normalize
    #
    # un_normalized_poses = np.multiply(normalized_poses, std) + mean
    #
    # # 2. Write
    # make_bvh_Trinity('output', 'pip_test_inference_py', loaded[0])
    # print('Finished!')


def scatter_plot(latent_representations: np.ndarray, labels: list) -> None:
    """The scatter plot for visualizing the latent representations with the ground truth class label.

    Args:
        latent_presentations: (N, dimension_latent_representation)
        labels: (N, )  the labels of the ground truth classes
    """

    # borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    # Note that if the dimension_latent_representation > 2 you need to apply TSNE transformation
    # to map the latent representations from higher dimensionality to 2D
    # You can use #from sklearn.manifold import TSNE#

    def discrete_cmap(n: int, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
        base = plt.cm.get_cmap(base_cmap)
        return base.from_list(base.name + str(n), base(np.linspace(0, 1, n)), n)

    plt.figure(figsize=(10, 10))
    plt.scatter(
        latent_representations[:, 0],
        latent_representations[:, 1],
        cmap=discrete_cmap(10, "jet"),
        c=labels,
        edgecolors="black",
    )
    plt.colorbar()
    plt.grid()
    plt.show()


def display_images_in_a_row(
    images: np.ndarray, file_path: str = "./tmp.png", display: bool = True
) -> None:
    """Save and/or display input images.

    Args:
        images: (N,28,28): N images of 28*28 as a numpy array
        file_path: file path name for where to store the figure
        display: display the image or not
    """
    save_image(images.view(-1, 1, 28, 28), "{}".format(file_path))
    if display is True:
        plt.imshow(mpimg.imread("{}".format(file_path)))
