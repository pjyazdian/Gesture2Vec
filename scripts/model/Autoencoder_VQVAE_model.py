"""This module contains models for Part b: Gesture Representation Learning.

The 'Autoencoder_VQVAE' model is the default model.
Certain model parameters should be included in a pre-specified config file.
See docstrings for the parameters that must be included for each model.

Based on the following Se2Seq implementations:
- https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
- https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

Typical usage example:
    args = config.parse_args()
    net = Autoencoder_VQVAE(args, 135, 30)
    result_tuple = net(in_tensor, one_start_frame_per_output_tensor)
"""


from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configargparse import argparse

debug = False


class EncoderRNN(nn.Module):
    """Custom RNN Encoder subclass.

    Attributes:
        input_size: The integer size of the input data.
        hidden_size: The integer size of the hidden layer in a Linear layer.
        n_layers: The integer size of GRU output layers.
        dropout: The float probability for adding noise to data.
        in_layer: A PyTorch Linear layer.
        gru: A PyTorch GRU layer.
        do_flatten_parameters: A boolean to flatten if GPU is available.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        dropout: float = 0.5,
        pre_trained_embedding: np.ndarray = None,
    ):
        """Initialize with input/hidden sizes, number of grus, dropout prob.

        Args:
            input_size: An integer size of the input data.
            hidden_size: An integer size of the hidden layer in a Linear layer.
            n_layers: An integer for the layers in GRU output size (default 1).
            dropout: A float probability for adding noise to data (default 50%).
            pre_trained_embedding: A word vector representation or None.
        """
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.in_layer = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True
        )
        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(
        self, input_seqs: torch.Tensor, hidden: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with input data tensor and hidden state tensor.

        Args:
            input_seqs: Tensor with shape (num_step(T),batch_size(B)),
                sorted decreasingly by lengths (for packing).
            input_seqs: A Tensor of sequence data to pass through.
            hidden: A Tensor of the initial state of GRU.

        Returns:
            A 2-Tuple:
                outputs: GRU outputs in shape (T,B,hidden_size(H))
                hidden: last hidden stat of RNN(i.e. last output for GRU)
        """
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()
        if debug:
            print("Encoder_Forward_input", input_seqs.shape)
        input_seq_in_layered = self.in_layer(input_seqs)
        outputs, hidden = self.gru(input_seq_in_layered, hidden)
        outputs: torch.Tensor = (
            outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        )  # Sum bidirectional outputs
        if debug:
            print("Encoder_Forward_ out, hidden", outputs.shape, hidden.shape)
        return outputs, hidden


class EncoderCNN(nn.Module):
    """Custom CNN Encoder subclass.

    Attributes:
        input_size: The integer size of the input data.
        hidden_size: The integer size of the hidden layer in a Linear layer.
        n_layers: The integer size of GRU output layers.
        dropout: The float probability for adding noise to data.
        in_layer: A PyTorch Linear layer.
        gru: A PyTorch GRU layer.
        cnn: A Sequential CNN model with Conv1d, BatchNorm1d and ReLu layers.
        do_flatten_parameters: A boolean to flatten if GPU is available.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        dropout: float = 0.5,
        pre_trained_embedding: np.ndarray = None,
    ):
        """Initialize with input/hidden sizes, number of grus, dropout prob.

        Args:
            input_size: An integer size of the input data.
            hidden_size: An integer size of the hidden layer in a Linear layer.
            n_layers: An integer for the layers in GRU output size (default 1).
            dropout: A float probability for adding noise to data (default 50%).
            pre_trained_embedding: A word vector representation or None.
        """
        super(EncoderCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.in_layer = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True
        )
        self.cnn = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, stride=2)
        channels = [
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
        ]
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=5,
                stride=5,
                bias=False,
            ),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=3,
                stride=3,
                bias=False,
            ),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=channels[2],
                out_channels=channels[3],
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(
        self, input_seqs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with input data tensor.

        Hidden state tensor is unneeded as this is a CNN.

        Args:
            input_seqs: A Tensor of shape (num_step(T),batch_size(B)),
                        sorted decreasingly by lengths (for packing).
            hidden: A Tensor of the initial state of GRU or None.

        Returns:
            outputs: GRU outputs in shape (T,B,hidden_size(H))
            hidden: Zero Tensor. This value is only for API consistency.
        """
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()
        if debug:
            print("Encoder_Forward_input", input_seqs.shape)
        input_seq_in_layered = self.in_layer(input_seqs)
        input_seq_in_layered = input_seq_in_layered.permute(
            (1, 2, 0)
        )  # 128,30,200 --> 30,200,128 n*c*l

        print(self.cnn)
        hidden = self.cnn(input_seq_in_layered)
        outputs = torch.tensor(0)
        if debug:
            print("Encoder_Forward_ out, hidden", outputs.shape, hidden.shape)
        return outputs, hidden


class DecoderCNN(nn.Module):
    """Custom CNN Decoder subclass.

    Attributes:
        input_size: The integer size of the input data.
        hidden_size: The integer size of the hidden layer in a Linear layer.
        n_layers: The integer size of GRU output layers.
        dropout: The float probability for adding noise to data.
        in_layer: A PyTorch Linear layer.
        gru: A PyTorch GRU layer.
        cnn: A Sequential CNN model with Conv1d, BatchNorm1d and ReLu layers.
        do_flatten_parameters: A boolean to flatten if GPU is available.
    """

    def __init__(
        self,
        input_size: torch.Tensor,
        hidden_size: int,
        n_layers: int = 1,
        dropout: float = 0.5,
        pre_trained_embedding: np.ndarray = None,
    ):
        """Initialize with input/hidden sizes, number of grus, dropout prob.

        Args:
            input_size: An integer size of the input data.
            hidden_size: An integer size of the hidden layer in a Linear layer.
            n_layers: An integer for the layers in GRU output size (default 1).
            dropout: A float probability to add noise to data (default 50%).
            pre_trained_embedding: A word vector representation or None.
        """
        super(DecoderCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.in_layer = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True
        )
        self.cnn = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, stride=2)
        channels = [
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
        ]
        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=3,
                stride=3,
                bias=False,
            ),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                in_channels=channels[2],
                out_channels=channels[3],
                kernel_size=5,
                stride=5,
                bias=False,
            ),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(
        self, input_seqs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with input data tensor.

        Hidden state tensor is unneeded as this is a CNN.

        Args:
            input_seqs: A Tensor of shape (num_step(T),batch_size(B)),
                sorted decreasingly by lengths(for packing)
            hidden: A Tensor of initial state of GRU or None.

        Returns:
            outputs: CNN outputs in shape (T,B,hidden_size(H))
            hidden: Zero Tensor. This value is only for API consistency.
        """
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()
        if debug:
            print("Encoder_Forward_input", input_seqs.shape)
            print(self.in_layer)
        input_seq_in_layered: torch.Tensor = self.in_layer(
            input_seqs.squeeze()
        ).unsqueeze(-1)
        input_seq_in_layered = input_seq_in_layered.unsqueeze(0)

        print("DCNN input_seq_in_layered", input_seq_in_layered.shape)

        outputs = self.cnn(input_seq_in_layered)
        hidden = torch.tensor(0)
        if debug:
            print("Encoder_Forward_ out, hidden", outputs.shape, hidden.shape)
        return outputs, hidden


class Attn(nn.Module):
    """Attention layer for scoring.

    Attributes:
        hidden_size: The integer size of the output (input is 2x this size).
        attn: A PyTorch Linear layer.
        v: A Tensor that can be the internal state at a specific point in time.
    """

    def __init__(self, hidden_size: int):
        """Initialize with the size of the hidden layer.

        Args:
            hidden_size: The interger size of the hidden layer.
        """
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with the hidden weights.

        Args:
            hidden: A Tensor of the previous hidden state of the decoder,
                in shape (layers*directions,B,H).
            encoder_outputs: A Tensor of encoder outputs from Encoder,
                in shape (T,B,H).

        Returns:
            A Tensor with attention energies in shape (B,T).
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Calculates energy score between 2 matrices (previous/current state).

        Args:
            hidden: A Tensor of the hidden state.
            encoder_outputs: A Tensor of encoder output.

        Returns:
            A Tensor with the energy score.
        """
        energy = torch.tanh(
            self.attn(torch.cat([hidden, encoder_outputs], 2))
        )  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class BahdanauAttnDecoderRNN(nn.Module):
    """Custom Decoder RNN class.

    Attributes:
        hidden_size: The integer size of the output of the hidden layer.
        output_size: The integer size of the output of the output layer.
        n_layers: An integer number of recurrent gru units.
        dropout_p: A float probability of adding noise in a dropout layer.
        embedding: A PyTorch Embedding lookup table for word vectors.
        dropout: A PyTorch Dropout layer with 'dropout_p' as the probability.
        pre_linear: A Linear layer with a BatchNorm1d and ReLU activation.
        attn: A custom 'Attn' object for calculating attention scores.
        gru: A PyTorch GRU RNN.
        out: A Linear layer applied to the gru output.
        do_flatten_parameters: A boolean to flatten if a GPU is available.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 1,
        dropout_p: float = 0.1,
        discrete_representation: bool = False,
        speaker_model=None,
    ):
        """Initialize with multiple parameters.

        The 'args' argument must have the following keys:
            autoencoder_conditioned: A string boolean to use zeroes or dropout.
            autoencoder_att: A string boolean to track 'Attn' scoring.
            autoencoder_fixed_weight: A string boolean to calculate gradients.

        Args:
            input_size: The integer size of input data.
            hidden_size: The integer size of the output of the hidden layer.
            output_size: The integer size of the output of the output layer.
            n_layers: A integer number of hidden layers.
            dropout_p: A float probability of adding noise in a dropout layer.
            discrete_representation: A boolean if language model is to be used.
            speaker_model: A 'Vocab' word vector representation.
        """
        super(BahdanauAttnDecoderRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.discrete_representation = discrete_representation
        self.speaker_model = speaker_model

        # define embedding layer
        if self.discrete_representation:
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.dropout = nn.Dropout(dropout_p)

        if args.autoencoder_conditioned == "True":
            self.autoencoder_conditioned = True
        else:
            self.autoencoder_conditioned = False

        # define layers
        if args.autoencoder_att == "True":
            self.attn = Attn(hidden_size)
            self.att_use = True
        else:
            self.att_use = False

        if self.att_use:
            linear_input_size = input_size + hidden_size
        else:
            linear_input_size = input_size
        self.pre_linear = nn.Sequential(
            nn.Linear(linear_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        if args.autoencoder_fixed_weight == "True":
            self.autoencoder_fixed_weight = True
            for param in self.gru.parameters():
                param.requires_grad = False

        self.out_layer = nn.Linear(hidden_size, output_size)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def freeze_attn(self) -> None:
        """Stop calculating gradients for attention layer."""
        for param in self.attn.parameters():
            param.requires_grad = False

    def forward(
        self,
        motion_input: torch.Tensor,
        last_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        vid_indices: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with motion input, previous decoder state and encoder output.

        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence

        Args:
            motion_input: A Tensor of motion input for current time step,
                in shape [batch x dim]
            last_hidden: A Tensor last hidden state of the decoder,
                in shape [layers x batch x hidden_size]
            encoder_outputs: A Tensor of encoder outputs,
                in shape [steps x batch x hidden_size]
            vid_indices (torch.Tensor): A Tensor of frame indices.

        Returns:
            A 3-Tuple:
                output: Tensor of output from decoder.
                decoder: Tensor of the hidden state.
                attn_weights: Tensor of custom 'Attn' layer weights.
        """

        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        if self.discrete_representation:
            word_embedded = self.embedding(motion_input).view(
                1, motion_input.size(0), -1
            )  # [1 x B x embedding_dim]
            motion_input = self.dropout(word_embedded)
        else:
            motion_input = motion_input.view(
                1, motion_input.size(0), -1
            )  # [1 x batch x dim]

        if debug:
            print("Decoder_forward, motion_input", motion_input.shape)  # [1, 128, 41])
        if debug:
            print(
                "Decoder_forward, encoder_outputs", encoder_outputs.shape
            )  # [1, 128, 41])

        # attention
        if self.att_use:
            attn_weights: torch.Tensor = self.attn(
                last_hidden[-1], encoder_outputs
            )  # [batch x 1 x T]
            context = attn_weights.bmm(
                encoder_outputs.transpose(0, 1)
            )  # [batch x 1 x attn_size]
            context = context.transpose(0, 1)  # [1 x batch x attn_size]

            # make input vec
            rnn_input = torch.cat(
                (motion_input, context), 2
            )  # [1 x batch x (dim + attn_size)]
        else:
            attn_weights = None
            rnn_input = motion_input
        if debug:
            print("Decoder_forward, rnn_input", rnn_input.shape)  # [1, 128, 241])

        # Check if unconditioned
        if self.autoencoder_conditioned == False:
            rnn_input = torch.zeros_like(rnn_input)
        rnn_input: torch.Tensor = nn.Dropout(0.95)(rnn_input)

        q = rnn_input.squeeze(0)
        rnn_input = self.pre_linear(rnn_input.squeeze(0))
        if debug:
            print(
                "Decoder_forward, rnn_input", rnn_input.shape
            )  # torch.Size([128, 200])

        rnn_input = rnn_input.unsqueeze(0)
        if debug:
            print(
                "Decoder_forward, rnn_input", rnn_input.shape
            )  # torch.Size([1, 128, 200])

        # rnn
        output, hidden = self.gru(rnn_input, last_hidden)

        # post-fc
        output = output.squeeze(0)  # [1 x batch x hidden_size] -> [batch x hidden_size]
        output = self.out_layer(output)

        return output, hidden, attn_weights


class Generator(nn.Module):
    """Custom RNN Decoder subclass.

    Attributes:
        output_size: An integer size of the output.
        n_layers: An integer number of hidden (recurrent) layers (in the RNN).
        discrete_representation: A boolean to use a word vector representation.
        decoder: A 'BahdanauAttnDecoderRNN' to use for generating output.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        motion_dim: int,
        discrete_representation: bool = False,
        speaker_model=None,
    ):
        """Initailize with prespecified parameters and input size.

        The 'args' argument must have the following keys:
            autoencoder_conditioned: A string boolean to use zeroes or dropout.
            autoencoder_att: A string boolean to track 'Attn' scoring.
            autoencoder_fixed_weight: A string boolean to calculate gradients.

        Args:
            args: A configargparser with prespecified parameters (See above).
            motion_dim: An integer dimension of the output data.
            discrete_representation: A boolean to use a word vector representation
                (default: false).
            speaker_model: A 'Vocab' pre-trained word vector representation.
        """
        super(Generator, self).__init__()
        self.output_size = motion_dim
        self.n_layers = args.n_layers
        self.discrete_representation = discrete_representation
        self.decoder = BahdanauAttnDecoderRNN(
            args=args,
            input_size=args.rep_learning_dim,
            hidden_size=args.hidden_size,
            output_size=args.rep_learning_dim,
            n_layers=self.n_layers,
            dropout_p=args.dropout_prob,
            discrete_representation=discrete_representation,
            speaker_model=speaker_model,
        )
        self.is_training = True

    def freeze_attn(self) -> None:
        """Stop calculating gradients for attention layer."""
        self.decoder.freeze_attn()

    def forward(
        self,
        z: torch.Tensor | None,
        motion_input: torch.Tensor,
        last_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        vid_indices: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with multiple Tensors of inputs and hidden weights.

        Argument 'z' is appended to each element in 'motion_input'.

        Args:
            z: A Tensor of noisy data or None.
            motion_input: A Tensor of input data.
            last_hidden: A Tensor of the previous hidden state.
            encoder_output: A Tensor of the output from encoder.
            vid_indices: A Tensor of frame indices or None. (unused).

        Returns:
            A 3-Tuple:
                output: Tensor of output from decoder.
                decoder: Tensor of the hidden state.
                attn_weights: Tensor of custom 'Attn' layer weights.
        """
        if z is None:
            input_with_noise_vec = motion_input
        else:
            assert (
                not self.discrete_representation
            )  # not valid for discrete representation
            input_with_noise_vec = torch.cat(
                [motion_input, z], dim=1
            )  # [bs x (10+z_size)]

        return self.decoder(
            input_with_noise_vec, last_hidden, encoder_output, vid_indices
        )


class Autoencoder_VQVAE(nn.Module):
    """Base model for gesture representation learning.

    Part b model as described in the paper. Can use CNN or RNN (default) as the
    encoder and decoder models. VAE and VQVAE models can be used in the latent
    code space. Some VAE/VQVAE models are experimental and are provided as-is.

    Attributes:
        encoder: A PyTorch model to encode gestures.
        out_layer_encoder: A Linear layer with Tanh activation to scale data.
        out_layer_decoder: A Linear layer to reshape output to input dimension.
        decoder: A PyTorch model to decode gestures.
        CNN: A boolean to use CNN (instead of RNN) as encoder and decoder.
        VAE: A boolean whether to use basic VAE.
        VAE_fc_mean: A Linear layer to process mean of encoder hidden state.
        VAE_fc_std: A Linear layer to process std of encoder hidden state.
        VAE_fc_decoder: A Linear layer to process data if VAE only.
        vq: A boolean whether to use VQVAE.
        vq_components: An integer number of clusters in VQVAE.
        commitment_cost: A float cost to include in loss calculations.
        vq_layer: A PyTorch custom VQVAE model.
        n_frames: An integer count of frames in a single sample.
        n_pre_poses: An integer count of frames as a starting point in sample.
        pose_dim: An integer dimension of a single sample.
        autoencoder_conditioned: A boolean to use initial frames or zeros.
        do: A Dropout layer with custom probability to add noise to data.
        autoencoder_fixed_encoder_weight: A boolean if encoder is frozen.
    """

    def __init__(self, args: argparse.Namespace, pose_dim: int, n_frames: int):
        """Initialize with parameters, the input dimension and frames.

        The 'args' argument must contain the following keys:
            rep_learning_dim: An integer size of the input data from encoder.
            hidden_size: An integer size of the output from hidden layer.
            n_layers: An integer for the layers in GRU output size.
            dropout_prob: A float probability for adding noise to data.
            autoencoder_vae: A string boolean whether to use basic VAE.
            autoencoder_vq: A string boolean whether to use VQVAE.
            autoencoder_vq_components: An integer number of clusters in VQVAE.
            autoencoder_vq_commitment_cost: A float cost modifier.
            n_pre_poses: An integer count of frames as a starting point.
            autoencoder_conditioned: A string boolean to use zeroes or dropout.
            autoencoder_att: A string boolean to track 'Attn' scoring.
            autoencoder_fixed_weight: A string boolean to calculate gradients.

        Args:
            args: An argparser object with specified parameters (See above).
            pose_dim: An integer dimension of the output data.
            n_frames: An integer number of frames in a output sequence.
        """
        super().__init__()

        self.CNN = False

        # 1. Defining Encoder and Decoder
        self.encoder = EncoderRNN(
            args.rep_learning_dim,
            args.hidden_size,
            args.n_layers,
            dropout=args.dropout_prob,
            pre_trained_embedding=None,
        )

        self.out_layer_encoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size), nn.Tanh()
        )
        self.out_layer_decoder = nn.Sequential(
            nn.Linear(args.hidden_size, pose_dim),
        )

        self.decoder = Generator(args, pose_dim, speaker_model=None)

        if self.CNN:
            self.encoder = EncoderCNN(
                args.rep_learning_dim,
                args.hidden_size,
                args.n_layers,
                dropout=args.dropout_prob,
                pre_trained_embedding=None,
            )
            self.decoder = DecoderCNN(
                args.hidden_size,
                args.hidden_size,
                args.n_layers,
                dropout=args.dropout_prob,
                pre_trained_embedding=None,
            )

        # Defining Variational Autoencoder layers
        if args.autoencoder_vae == "True":
            self.VAE = True
            self.VAE_fc_mean = nn.Linear(
                self.decoder.n_layers * args.hidden_size,
                self.decoder.n_layers * args.hidden_size,
            )
            self.VAE_fc_std = nn.Linear(
                self.decoder.n_layers * args.hidden_size,
                self.decoder.n_layers * args.hidden_size,
            )
            self.VAE_fc_decoder = nn.Linear(
                self.decoder.n_layers * args.hidden_size,
                self.decoder.n_layers * args.hidden_size,
            )
        else:
            self.VAE = False

        # 2. Defining  Vector Quantizer layer
        if args.autoencoder_vq == "True":
            decay = 0.85
            self.vq = True
            self.vq_components = int(args.autoencoder_vq_components)
            self.commitment_cost = float(args.autoencoder_vq_commitment_cost)
            # Todo: add a pre linear layer before feeding to the VQ layer

            if decay > 0:
                self.vq_layer = VQ_Payam_EMA(
                    self.vq_components,
                    args.hidden_size * args.n_layers,
                    self.commitment_cost,
                    decay,
                )

            else:
                self.vq_layer = VQ_Payam(
                    self.vq_components,
                    args.hidden_size * args.n_layers,
                    self.commitment_cost,
                )

            self.vq_layer = VQ_Payam_GSSoft(
                self.vq_components,
                args.hidden_size * args.n_layers,
                self.commitment_cost,
            )

            # self.vq_layer  = VectorQuantGroup(n_channels=1,
            #                                   n_classes=512,
            #                                   vec_len=30,
            #                                   num_group=16,
            #                                   num_sample=32,
            #                                   normalize=False)

        else:
            self.vq = False

        # 3. Define parameters
        self.n_frames = n_frames
        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = args.rep_learning_dim

        if args.autoencoder_conditioned == "True":
            self.autoencoder_conditioned = True
        else:
            self.autoencoder_conditioned = False

        self.do = nn.Dropout(args.dropout_prob)

        # Todo: should fix, currently we freez from train_epoch function
        # if args.autoencoder_freeze_encoder=='True':
        #     self.freez_encoder()

    def freez_encoder(self) -> None:
        """Freezes all encoder weights by stopping gradient calculations.

        Set parameters.requires_grad to False for the following layers/modules:
            - encoder
            - VAE_fc_mean
            - VAE_fc_std
            - VAE_fc_decoder
        """
        self.autoencoder_fixed_encoder_weight = True

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.VAE_fc_mean.parameters():
            param.requires_grad = False
        for param in self.VAE_fc_std.parameters():
            param.requires_grad = False
        for param in self.VAE_fc_decoder.parameters():
            param.requires_grad = False

    def freeze_VQminuse(self) -> None:
        """Freezes all encoder and decoder weights."""
        if self.VAE:
            self.freez_encoder()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False

    def reparameterize(
        self, mu: torch.Tensor, logVar: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        """Add noise to 'mu' input if training.

        Args:
            mu: A Tensor of input data.
            logVar: A Tensor of variances.
            train: A boolean whether to add noise or not.

        Returns:
            A Tensor of noisy data or the original 'mu' input.
        """
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        # Todo: fix this for train and test time checking
        if train:
            return mu + std * eps
        else:
            return mu  # + std * eps

    def forward(
        self,
        in_poses: torch.Tensor,
        out_poses: torch.Tensor,
        vq_layer_active: bool = False,
    ) -> (
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor]
    ):
        """Forward pass with input and output starting point data and if VQVAE.

        Note that 'in_poses'/'out_poses' are transposed (0,1) as needed.

        Args:
            in_poses: A Tensor of input data.
            out_poses: A Tensor of initial data to generate output based on.
            vq_layer_active: A boolean if VQVAE is being used.

        Returns:
            A 2- or 4- or 6-Tuple as follows:
                case 1 (6-Tuple) - self.vq is True and self.vae is True:
                    outputs: A Tensor of output data from decoder.
                    decoder_first_hidden: A Tensor of pure hidden state.
                    mean: A Tensor of hidden state used as mean (VAE_fc_mean).
                    logVar: A Tensor of hidden state used as std (VAE_fc_std).
                    loss_vq: A Tensor of loss values (MSELoss).
                    perplexity_vq: A Tensor of perplexity of the latent space.
                case 2 (4-Tuple) - self.vq is True and self.vae is False:
                    outputs: Same as above.
                    decoder_first_hidden: Same as above.
                    loss_vq: Same as above.
                    perplexity_vq: Same as above.
                case 3 (4-Tuple) - self.vq is False and self.vae is True:
                    outputs: Same as above.
                    decoder_first_hidden: Same as above.
                    mean: Same as above.
                    logVar: Same as above.
                case 4 (2-Tuple) - self.vq is False and self.vae is False:
                    outputs: Same as above.
                    decoder_first_hidden: Same as above.
        """

        # if self.vq and vq_layer_active:
        #     self.freeze_VQminuse()

        # reshape to (seq x batch x dim)
        # in_text = in_text.transpose(0, 1)
        in_poses = in_poses.transpose(0, 1)
        in_poses = self.do(in_poses)
        out_poses = out_poses.transpose(0, 1)

        if not self.CNN:
            outputs = torch.zeros(
                self.n_frames, out_poses.size(1), self.decoder.output_size
            ).to(out_poses.device)

        # run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(in_poses, None)

        if self.CNN:
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = encoder_hidden[
                : self.decoder.n_layers
            ]  # use last hidden state from encoder
        # Todo: I put vector quantization here. However, we need to check it using both VAE and not VAE.
        vq_layer_active = True
        if self.vq and vq_layer_active:
            loss_vq, quantized, perplexity_vq, encodings = self.vq_layer(decoder_hidden)
            decoder_hidden = quantized
        else:
            loss_vq, quantized, perplexity_vq, encodings = (
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
            )

        # print("Sanity check:", encoder_hidden[:self.decoder.n_layers].shape)
        # self.VAE = False
        if self.VAE:
            if debug:
                print("self.VAE", self.VAE)
            # decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder
            # [2, 128, 200]
            # print("decoder_hidden!!! org", decoder_hidden.shape)
            decoder_hidden = decoder_hidden.transpose(
                1, 0
            ).contiguous()  # [128, 2, 200]
            decoder_hidden = torch.reshape(
                decoder_hidden, (decoder_hidden.shape[0], -1)
            )
            mean = self.VAE_fc_mean(decoder_hidden)
            logvar = self.VAE_fc_std(decoder_hidden)
            z = self.reparameterize(mean, logvar)
            z = self.VAE_fc_decoder(z)
            decoder_hidden = z.reshape(
                decoder_hidden.shape[0], self.decoder.n_layers, -1
            )
            decoder_hidden = decoder_hidden.transpose(1, 0).contiguous()
            # print("decoder_hidden!!! modified", decoder_hidden.shape)
            decoder_first_hidden = decoder_hidden
        # else:
        #     decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder
        #     # print("decoder_hidden!!! not VAE ", decoder_hidden.shape)

        if debug:
            print("Decoder hidden from encoder:", decoder_hidden.shape)

        decoder_first_hidden = decoder_hidden

        # Todo: should be fixed such a parameter for inference time
        all_hiddens = []

        if self.CNN:
            outputs, hidden = self.decoder(decoder_hidden)
            shape = outputs.shape
            # print(shape)
            outputs = outputs.transpose(1, 2)
            # print(outputs.shape)
            shape = outputs.shape
            outputs = outputs.reshape(shape[0] * shape[1], shape[2])
            # print(outputs.shape)
            outputs = self.out_layer_decoder(outputs)
            outputs = outputs.reshape(shape[0], shape[1], -1)
            outputs = outputs.transpose(0, 1)
            # print(outputs.shape)

        else:
            # run through decoder one time step at a time
            decoder_input = out_poses[0]  # initial pose from the dataset
            outputs[0] = decoder_input
            for t in range(1, self.n_frames):
                if not self.autoencoder_conditioned:
                    decoder_input = torch.zeros_like(decoder_input)
                decoder_output, decoder_hidden, _ = self.decoder(
                    None, decoder_input, decoder_hidden, encoder_outputs, None
                )
                outputs[t] = decoder_output

                if t < self.n_pre_poses:
                    decoder_input = out_poses[t]  # next input is current target
                else:
                    decoder_input = decoder_output  # next input is current prediction
                if not self.autoencoder_conditioned:
                    decoder_input = torch.zeros_like(decoder_input)

        if self.vq:
            if self.VAE:
                return (
                    outputs.transpose(0, 1),
                    decoder_first_hidden[: self.decoder.n_layers],
                    mean,
                    logvar,
                    loss_vq,
                    perplexity_vq,
                )
            else:
                return (
                    outputs.transpose(0, 1),
                    decoder_first_hidden[: self.decoder.n_layers],
                    loss_vq,
                    perplexity_vq,
                )
        else:
            if self.VAE:
                return (
                    outputs.transpose(0, 1),
                    decoder_first_hidden[: self.decoder.n_layers],
                    mean,
                    logvar,
                )
            else:
                return (
                    outputs.transpose(0, 1),
                    decoder_first_hidden[: self.decoder.n_layers],
                )


class VQ_Payam(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """ """
        super(VQ_Payam, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_linear = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._embedding.weight.data.uniform_(-0.2, 0.2)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        # self.embedding_grad(False)

    def embedding_grad(self, what: bool) -> None:
        """ """
        for param in self._embedding.parameters():
            param.requires_grad = what

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """

        loss_vq, loss, perplexity_vq, encodings = (
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        )
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        # flat_input = self.pre_linear(flat_input)

        # __________________________
        # 2. Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # 3. Find nearest encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # print("Idices: ", encoding_indices)

        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # 4. Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight
        )  # .view(input_shape)

        m = torch.mean(quantized)
        mw = torch.std(quantized)

        quantized = torch.reshape(quantized, inputs.shape).contiguous()

        torch.mean(quantized)

        # 5. Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # loss = torch.square(loss)

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.contiguous(), perplexity, encodings
        # ___________________________

        quantized = torch.reshape(flat_input, inputs.shape).contiguous()

        return loss, quantized, perplexity_vq, encodings


class VQ_Payam_EMA(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float = 1e-5,
    ):
        """ """
        super(VQ_Payam_EMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_linear = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._embedding.weight.data.uniform_(-1, 1)
        # self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        #     EMA
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """

        loss_vq, loss, perplexity_vq, encodings = (
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        )
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        flat_input = self.pre_linear(flat_input)

        # __________________________
        # 2. Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # 3. Find nearest encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # print("Idices: ", encoding_indices)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Todo:
        # Test modification for soft vq
        # ...

        # 4. Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight
        )  # .view(input_shape)
        quantized = torch.reshape(quantized, inputs.shape).contiguous()

        # 5.1 Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # 5. Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        loss = self._commitment_cost * e_latent_loss
        # loss = torch.square(loss)

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.contiguous(), perplexity, encodings
        # ___________________________

        quantized = torch.reshape(flat_input, inputs.shape).contiguous()

        return loss, quantized, perplexity_vq, encodings


class VQ_Payam_GSSoft(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """ """
        super(VQ_Payam_GSSoft, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_linear = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        # self._embedding.weight.data.uniform_(-0.2, 0.2)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        # self.embedding_grad(False)

        self.mean_layer = nn.Linear(embedding_dim, embedding_dim)
        self.logvar_layer = nn.Linear(embedding_dim, num_embeddings)

    def embedding_grad(self, what: bool) -> None:
        """ """
        for param in self._embedding.parameters():
            param.requires_grad = what

    def reparameterize(
        self, mu: torch.Tensor, logVar: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        """ """
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        # Todo: fix this for train and test time checking
        if train:
            return mu + std * eps
        else:
            return mu  # + std * eps

    def soft_prob(self, dist: float, smooth: float) -> float:
        """ """
        dist = (dist) / 400
        # print("Dist", dist)
        # print("Smooth", smooth)
        # zarb = torch.multiply(dist, 0.5 * smooth)
        # print("torch.multiply(dist, 0.5 * smooth)", zarb)
        # print(torch.exp(-zarb))
        # print(dist.shape, smooth.shape)

        # print((-torch.multiply(dist, 0.5 * smooth)))
        # print(dist)
        prob = torch.exp(-torch.multiply(dist, 0.5 * smooth)) / torch.sqrt(smooth)

        # print(prob)
        # sumprobsumprob = prob.sum(1)
        # print("sum_pron", sumprobsumprob.shape, prob.shape)
        # print(prob.sum(1).unsqueeze(1).expand(prob.shape[0], prob.shape[1]).shape)
        # print("Mean", prob.sum(0))
        probs = prob / prob.sum(1).unsqueeze(1).expand(prob.shape[0], prob.shape[1])
        # print(probs.shape)
        # print("Normalized Prob", probs)

        return probs

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """

        loss_vq, loss, perplexity_vq, encodings = (
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        )
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        # flat_input = self.pre_linear(flat_input)

        flat_input = self.mean_layer(flat_input)
        z_logvar = self.logvar_layer(flat_input)

        # __________________________
        # 2. Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # # 3. Find nearest encoding
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # # print("Idices: ", encoding_indices)
        # encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # encodings.scatter_(1, encoding_indices, 1)

        smooth = 1.0 / torch.exp(z_logvar) ** 2
        probs = self.soft_prob(distances, smooth)
        encodings = probs
        # print("---1--", probs.shape, probs)

        # 4. Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight
        )  # .view(input_shape)

        quantized = torch.reshape(quantized, inputs.shape).contiguous()

        # 5. Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # loss = torch.square(loss)

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.contiguous(), perplexity, encodings
        # ___________________________

        quantized = torch.reshape(flat_input, inputs.shape).contiguous()

        return loss, quantized, perplexity_vq, encodings


class VQ_Payam_GSSoft16(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """ """
        super(VQ_Payam_GSSoft16, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.reduced_dim = 16
        self.pre_linear = nn.Linear(self._embedding_dim, self.reduced_dim)
        self.post_linear = nn.Linear(self.reduced_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self.reduced_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        # self._embedding.weight.data.uniform_(-0.2, 0.2)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        # self.embedding_grad(False)

        self.mean_layer = nn.Linear(self.reduced_dim, self.reduced_dim)
        self.logvar_layer = nn.Linear(self.reduced_dim, num_embeddings)

    def embedding_grad(self, what: bool) -> None:
        """ """
        for param in self._embedding.parameters():
            param.requires_grad = what

    def reparameterize(
        self, mu: torch.Tensor, logVar: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        """ """

        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        # Todo: fix this for train and test time checking
        if train:
            return mu + std * eps
        else:
            return mu  # + std * eps

    def soft_prob(self, dist: float, smooth: float) -> float:
        dist = (dist) / 400
        # print("Dist", dist)
        # print("Smooth", smooth)
        # zarb = torch.multiply(dist, 0.5 * smooth)
        # print("torch.multiply(dist, 0.5 * smooth)", zarb)
        # print(torch.exp(-zarb))
        # print(dist.shape, smooth.shape)

        # print((-torch.multiply(dist, 0.5 * smooth)))
        # print(dist)
        prob = torch.exp(-torch.multiply(dist, 0.5 * smooth)) / torch.sqrt(smooth)

        # print(prob)
        # sumprobsumprob = prob.sum(1)
        # print("sum_pron", sumprobsumprob.shape, prob.shape)
        # print(prob.sum(1).unsqueeze(1).expand(prob.shape[0], prob.shape[1]).shape)
        # print("Mean", prob.sum(0))
        probs = prob / prob.sum(1).unsqueeze(1).expand(prob.shape[0], prob.shape[1])
        # print(probs.shape)
        # print("Normalized Prob", probs)

        return probs

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """

        loss_vq, loss, perplexity_vq, encodings = (
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        )
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        flat_input = self.pre_linear(flat_input)

        flat_input = self.mean_layer(flat_input)
        z_logvar = self.logvar_layer(flat_input)

        # __________________________
        # 2. Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # # 3. Find nearest encoding
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # # print("Idices: ", encoding_indices)
        # encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # encodings.scatter_(1, encoding_indices, 1)

        smooth = 1.0 / torch.exp(z_logvar) ** 2
        probs = self.soft_prob(distances, smooth)
        encodings = probs
        # print("---1--", probs.shape, probs)

        # 4. Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight
        )  # .view(input_shape)

        quantized = self.post_linear(quantized)
        quantized = torch.reshape(quantized, inputs.shape).contiguous()

        # 5. Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # loss = torch.square(loss)

        quantized = inputs + (quantized - inputs).detach()

        encodings_HARD = encodings.argmax(dim=1)
        avg_probs = torch.mean(encodings, dim=0)

        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.contiguous(), perplexity, encodings
        # ___________________________

        quantized = torch.reshape(flat_input, inputs.shape).contiguous()

        return loss, quantized, perplexity_vq, encodings


class VectorQuantizer(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """ """
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_lin = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        if debug:
            print("VQ_Embedding", self._embedding)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """

        loss_vq, loss, perplexity_vq, encodings = (
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        )
        return loss, inputs, perplexity_vq, encodings

        org = inputs.detach().clone()
        # convert inputs from BCHW -> BHWC
        # Convert input from Layers, Batch, HDim -> Batch, Layers.HDim
        # Convert input from 2     , 128  , 200 -> 128   , 400
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # Todo: fix this to n layers (currently for two layers)
        first_shape = inputs.shape
        inputs = torch.hstack((inputs[0], inputs[1]))
        input_shape = inputs.shape
        if debug:
            print("VQ input_shape:", input_shape)

        # inputs = self.pre_lin(inputs)
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        if debug:
            print("VQ_flat_input", flat_input.shape)

        loss_vq, loss, perplexity_vq, encodings = (
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        )
        quantized = torch.reshape(inputs, first_shape).contiguous()

        xxx = torch.sum(org - quantized)

        return loss, quantized, perplexity_vq, encodings

        # Calculate distances
        if debug:
            print(
                "(torch.sum(flat_input ** 2, dim=1, keepdim=True)",
                torch.sum(flat_input**2, dim=1, keepdim=True).shape,
            )
            print(
                "torch.sum(self._embedding.weight ** 2, dim=1)",
                torch.sum(self._embedding.weight**2, dim=1).shape,
            )
            print(
                "2 * torch.matmul(flat_input, self._embedding.weight.t()))",
                (2 * torch.matmul(flat_input, self._embedding.weight.t())).shape,
            )
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # print("Idices: ", encoding_indices)

        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Todo: to sanity check
        # loss = 0
        # quantized = inputs

        # convert quantized from BHWC -> BCHW

        # from Layers, Batch, HDim -> Batch, Layers.HDim
        # quantized.permute(0, 3, 1, 2)
        # Convert input from 2     , 128  , 200 -> 128   , 400
        if debug:
            print("VQ_quantized:", quantized.shape)
        # Todo: is it ok to reshape like this.
        quantized = torch.reshape(quantized, (2, quantized.shape[0], -1)).contiguous()
        if debug:
            print("VQ_quantized:", quantized.shape)

        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float = 1e-5,
    ):
        """ """
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_lin = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = torch.hstack((inputs[0], inputs[1]))
        input_shape = inputs.shape

        inputs = self.pre_lin(inputs)

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW

        quantized = torch.reshape(quantized, (2, quantized.shape[0], -1)).contiguous()

        return loss, quantized, perplexity, encodings


class VectorQuantGroup(nn.Module):
    """
    This class documentation is #TODO.

        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        vec_len: int,
        num_group: int,
        num_sample: int,
        normalize: bool = False,
    ):
        """ """
        super().__init__()
        self._commitment_cost = 0.25
        if normalize:
            target_scale = 0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3
            self.normalize_scale = None

        self.n_classes = n_classes
        self._num_group = num_group
        self._num_sample = num_sample
        if not self.n_classes % self._num_group == 0:
            raise ValueError("num of embeddings in each group should be an integer")
        self._num_classes_per_group = int(self.n_classes / self._num_group)

        # self.embedding0 = nn.Parameter(
        #     torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self._num_embeddings = n_classes
        self._embedding_dim = 400
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.after_update()

    def forward(
        self, x0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = (
                target_norm
                * self._embedding
                / self._embedding.norm(dim=2, keepdim=True)
            )
        else:
            x = x0
            embedding = self._embedding
        # logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(-1, self._embedding_dim)
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        prob_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            # d = (x1_chunk - embedding).norm(dim=3)
            d = (
                torch.sum(x1**2, dim=1, keepdim=True)
                + torch.sum(embedding.weight**2, dim=1)
                - 2 * torch.matmul(x1, embedding.weight.t())
            )

            # Compute the group-wise distance
            d_group = torch.zeros(x1_chunk.shape[0], self._num_group).to(
                torch.device("cuda")
            )
            for i in range(self._num_group):
                d_group[:, i] = torch.mean(
                    d[
                        :,
                        i
                        * self._num_classes_per_group : (i + 1)
                        * self._num_classes_per_group,
                    ],
                    1,
                )
            degrup_numpy = d_group.detach().cpu().numpy()
            # Find the nearest group
            index_chunk_group = d_group.argmin(dim=1).unsqueeze(1)

            # Generate mask for the nearest group
            index_chunk_group = index_chunk_group.repeat(1, self._num_classes_per_group)
            index_chunk_group = torch.mul(
                self._num_classes_per_group, index_chunk_group
            )
            idx_mtx = (
                torch.LongTensor([x for x in range(self._num_classes_per_group)])
                .unsqueeze(0)
                .cuda()
            )
            index_chunk_group += idx_mtx
            showindex = index_chunk_group.detach().cpu().numpy()
            encoding_mask = torch.zeros(x1_chunk.shape[0], self.n_classes).cuda()
            encoding_mask.scatter_(1, index_chunk_group, 1)
            showencoding = encoding_mask.detach().cpu().numpy()

            # Todo: fixed up to here!

            # Compute the weight atoms in the group
            # Comment: Why should we use the squeeze()
            encoding_prob = torch.div(1, d.squeeze())

            # Apply the mask
            masked_encoding_prob = torch.mul(encoding_mask, encoding_prob)
            p, idx = masked_encoding_prob.sort(dim=1, descending=True)
            prob_chunks.append(p[:, : self._num_sample])
            index_chunks.append(idx[:, : self._num_sample])

        index = torch.cat(index_chunks, dim=0)
        prob_dist = torch.cat(prob_chunks, dim=0)
        prob_dist = F.normalize(prob_dist, p=1, dim=1)
        # index: (N*samples, n_channels) long tensor
        if True:  # compute the entropy
            hist = (
                index[:, 0]
                .float()
                .cpu()
                .histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            )
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = -(prob * prob.log()).sum().item()
            # logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        entropy = torch.tensor(entropy)
        index1 = index + self.offset
        # index1: (N*samples*n_channels) long tensor
        output_list = []
        for i in range(self._num_sample):
            output_list.append(
                torch.mul(
                    embedding.weight.view(-1, embedding.weight.size(1)).index_select(
                        dim=0, index=index1[:, i]
                    ),
                    prob_dist[:, i].unsqueeze(1).detach(),
                )
            )

        output_cat = torch.stack(output_list, dim=2)
        output_flat = torch.sum(output_cat, dim=2)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = F.mse_loss(x.detach(), output)  # .float().norm(dim=2).pow(2)
        out2 = F.mse_loss(
            x, output.detach()
        )  # .float()) #.norm(dim=2).pow(2) + (x - x0).float().norm(dim=2).pow(2)
        # logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        # return (out0, out1, out2, entropy)

        quantized = out0
        e_latent_loss = out2.mean()
        q_latent_loss = out1.mean()
        # Loss
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        perplexity = entropy
        encodings = encoding_prob.unsqueeze(0)
        return loss, quantized, perplexity, encodings

        discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
        # discrete: (N, 14, 1, 64)

    def after_update(self) -> None:
        """ """
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self._embedding.size(2))
                self._embedding.mul_(
                    target_norm / self._embedding.norm(dim=2, keepdim=True)
                )
