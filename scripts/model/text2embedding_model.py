"""Part c: Gestures Sequence Chunking and Part d: Text to Gesture Translation

Based on the following Se2Seq implementations:
- https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
- https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
"""


from __future__ import annotations
import math
import random
from typing import Tuple
from configargparse import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
from model.Helper_models import EncoderRNN_With_Audio, TextEncoderTCN
from model.vocab import Vocab

debug = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    """Custom RNN Encoder subclass.

    Attributes:
        input_size: The integer size of the input data.
        embed_size: The integer dimension of the word vector representation.
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
        embed_size: int,
        hidden_size: int,
        n_layers: int = 1,
        dropout: float = 0.5,
        pre_trained_embedding: np.ndarray = None,
    ):
        """Initialize with input/hidden sizes, number of grus, dropout prob.

        Args:
            input_size: An integer size of the input data.
            embed_size: An integer dimension of the word vector representation.
            hidden_size: An integer size of the hidden layer in a Linear layer.
            n_layers: An integer for the layers in GRU output size (default 1).
            dropout: A float probability for adding noise to data (default 50%).
            pre_trained_embedding: A word vector representation or None.
        """
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout

        # use pre-trained embedding (e.g., word2vec, glove)
        if pre_trained_embedding is not None:
            assert pre_trained_embedding.shape[0] == input_size
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pre_trained_embedding), freeze=False
            )
        else:
            self.embedding = nn.Embedding(input_size, embed_size)

        self.gru = nn.GRU(
            embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(
        self,
        input_seqs: torch.Tensor,
        input_lengths: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with input data tensor and hidden state tensor.

        Args:
            input_seqs: Tensor with shape (num_step(T),batch_size(B)),
                        sorted decreasingly by lengths (for packing).
            input_lengths: A Tensor of sequence data to pass through.
            hidden: A Tensor of the initial state of GRU.

        Returns:
            A 2-Tuple:
                outputs: GRU outputs in shape (T,B,hidden_size(H))
                hidden: last hidden stat of RNN(i.e. last output for GRU)
        """
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            outputs
        )  # unpack (back to padded)
        outputs = (
            outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        )  # Sum bidirectional outputs
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
        """Forward pass with the hidden weights and encoder output.

        Args:
            hidden: A Tensor of the previous hidden state of the decoder,
                    in shape (layers*directions,B,H).
            encoder_outputs: A Tensor of output from Encoder, in shape (T,B,H).

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
            encoder_outputs: A Tensor of output from encoder.

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
        speaker_model: Vocab | None = None,
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
            discrete_representation: A boolean if a language model is used.
            speaker_model: A 'Vocab' word vector representation or None.
        """
        super(BahdanauAttnDecoderRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size
        if noisy:
            self.hidden_size = hidden_size * 2
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.discrete_representation = discrete_representation
        self.speaker_model = speaker_model

        # define embedding layer
        if self.discrete_representation:
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.dropout = nn.Dropout(0.5)  # (dropout_p)

        if self.speaker_model:
            self.speaker_embedding = nn.Embedding(speaker_model.n_words, 8)

        # calc input size
        if self.discrete_representation:
            input_size = hidden_size  # embedding size
        linear_input_size = input_size + hidden_size
        if self.speaker_model:
            linear_input_size += 8

        # define layers

        if args.autoencoder_att == "True":
            self.attn = Attn(hidden_size)
            self.att_use = True
        else:
            self.att_use = False

        if self.att_use:
            linear_input_size = input_size + hidden_size
            self.attn = Attn(hidden_size)
        else:
            linear_input_size = input_size

        self.pre_linear = nn.Sequential(
            nn.Linear(linear_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)

        # self.out = nn.Linear(hidden_size * 2, output_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

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
        vid_indices: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with motion input, previous decoder state and encoder output.

        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence

        Args:
            motion_input: A Tensor of motion input for current time step,
                in shape [batch x dim]
            last_hidden: A Tensor last hidden state of the decoder,
                in shape [layers x batch x hidden_size]
            encoder_outputs: A Tensor of encoder outputs
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

        # attention

        if self.att_use:
            attn_weights = self.attn(
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
            # make input vec
            # rnn_input = torch.cat((motion_input, encoder_outputs), 2)  # [1 x batch x (dim + attn_size)]
            rnn_input = motion_input

        for_check = rnn_input.detach().cpu().numpy()
        if self.speaker_model:
            assert vid_indices is not None
            speaker_context = self.speaker_embedding(vid_indices).unsqueeze(0)
            rnn_input = torch.cat(
                (rnn_input, speaker_context), 2
            )  # [1 x batch x (dim + attn_size + embed_size)]

        rnn_input = self.pre_linear(rnn_input.squeeze(0))
        rnn_input = rnn_input.unsqueeze(0)

        # rnn
        output, hidden = self.gru(rnn_input, last_hidden)

        if debug:
            print("rnn_input", rnn_input.shape)
            print("last_hidden", last_hidden.shape)
            print(output.shape)
        # post-fc

        output = output.squeeze(0)  # [1 x batch x hidden_size] -> [batch x hidden_size]
        output = self.out((output))
        a = output.cpu().detach().numpy()
        # output = self.softmax(output)
        b = output.cpu().detach().numpy()
        # output = self.softmax(output)
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
        speaker_model: Vocab | None = None,
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
            args,
            input_size=motion_dim,
            hidden_size=args.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            dropout_p=args.dropout_prob,
            discrete_representation=discrete_representation,
            speaker_model=speaker_model,
        )

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
            vid_indices: A Tensor of frame indices or None.

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


noisy = False
audio_context = False
use_TCN = True
GPT3_embedding_active = False


class text2embedding_model(nn.Module):
    """Base model for text to gesture generation.

    Attributes:
        text2_embedding_discrete: A boolean to use word vector representation.
        n_layers: #TODO.
        encoder: A custom RNN encoder model with/without audio data for text.
        decoder: A custom RNN decoder model for gestures.
        n_frames: An integer count of frames in a single sample.
        n_pre_poses: An integer of frames as a starting point in gesture sample.
        pose_dim: An integer dimension of a single gesture sample.
        sentence_frame_length: An integer of frames for a text sample.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        pose_dim: int,
        n_frames: int,
        n_words: int,
        word_embed_size: int,
        word_embeddings: np.ndarray,
        speaker_model: Vocab | None = None,
    ):
        """Initialize with multiple parameters.

        The 'args' argument must have the following keys:
            text2_embedding_discrete: A string boolean to use word vector rep.
            n_layers: #TODO.
            autoencoder_vq_components: An integer number of clusters in VQVAE.
            hidden_size: An integer number of output of hidden Linear layer.
            dropout_prob: A float probability to add noise to data.
            autoencoder_att: A string boolean to track 'Attn' scoring.
            n_pre_poses: An integer of frames as a starting point in sample.
            sentence_frame_length: An integer of frames for a sample of text.

        Args:
            args: A configargparser object with specified values (See above).
            pose_dim: An integer dimension of the output data (gesture).
            n_frames: An integer of frames in a single sample (gesture).
            n_words: An integer size of the input data.
            word_embed_size: An integer dimension of the word vector rep.
            word_embeddings: A pre-trained word vector representation.
            speaker_model: A custom 'Vocab' object of word vector rep or None.
        """
        super().__init__()

        if args.text2_embedding_discrete == "True":
            self.text2_embedding_discrete = True
        else:
            self.text2_embedding_discrete = False

        self.n_layers = args.n_layers
        # Todo: find a better way to play with representation learning dimension
        # Todo: Regarding hidden size, sentence time length, etc.
        if self.text2_embedding_discrete:
            pose_dim = int(args.autoencoder_vq_components)
        else:
            pose_dim = args.n_layers * args.hidden_size

        self.encoder = EncoderRNN(
            n_words,
            word_embed_size,
            args.hidden_size,
            args.n_layers,
            dropout=args.dropout_prob,
            pre_trained_embedding=word_embeddings,
        )
        self.decoder = Generator(
            args,
            pose_dim,
            discrete_representation=self.text2_embedding_discrete,
            speaker_model=speaker_model,
        )

        if audio_context:
            self.encoder = EncoderRNN_With_Audio(
                n_words,
                word_embed_size,
                args.hidden_size,
                args.n_layers,
                dropout=args.dropout_prob,
                pre_trained_embedding=word_embeddings,
            )

        if use_TCN:
            self.encoder = TextEncoderTCN(
                args,
                n_words,
                embed_size=300,
                pre_trained_embedding=word_embeddings,
                kernel_size=2,
                dropout=0.3,
                emb_dropout=0.1,
            )

        if GPT3_embedding_active:
            self.encoder = DNN(
                n_layers=5,
                hidden_units=1024,
                input_dim=1024,
                output_dim=args.hidden_size * self.n_layers,
                device=device,
            )

            args.autoencoder_att = "False"
            self.decoder = Generator(
                args,
                pose_dim,
                discrete_representation=self.text2_embedding_discrete,
                speaker_model=speaker_model,
            )

        self.n_frames = n_frames
        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = pose_dim
        self.sentence_frame_length = args.sentence_frame_length

    def forward(
        self,
        in_text: torch.Tensor,
        in_lengths: torch.Tensor,
        in_audio: torch.Tensor,
        poses: torch.Tensor,
        GPT3_embeddings: torch.Tensor,
        vid_indices: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass with multiple parameters.

        Args:
            in_text: A Tensor of input text data as word vector representations.
            in_lengths: A Tensor of dimensions of the arg in_text.
            in_audio: A Tensor of audio data.
            poses: A Tensor of gesture data to use as starting point in output.
            GPT3_embeddings: A Tensor of word vectors data from GPT3.
            vid_indices: A Tensor of frame indices or None.

        Returns:
            A 2-Tuple:
                outputs: A 2D Tensor of output data.
                attentions_list: A list of Tensors of 'Attn' scores.
        """
        # reshape to (seq x batch x dim)
        in_text = in_text.transpose(0, 1)
        poses = poses.transpose(0, 1)

        outputs = torch.zeros(
            self.sentence_frame_length // self.n_frames,
            poses.size(1),
            self.decoder.output_size,
        ).to(poses.device)

        # if debug:
        # print("outputs", outputs.shape, "poses.shape", poses.shape)
        # print((outputs[2,0,:]))
        if self.text2_embedding_discrete:
            onehottie = F.one_hot(
                poses.reshape(-1).to(torch.int64), self.pose_dim
            ).reshape(poses.shape[0], poses.shape[1], -1)
        # print(onehottie[0])

        # outputs = onehottie.float()

        # run words through encoder
        if GPT3_embedding_active:
            encoder_outputs, encoder_hidden = None, self.encoder(GPT3_embeddings)
            #     Reshape required
            encoder_hidden = encoder_hidden.view(
                (self.n_layers, -1, encoder_hidden.shape[-1] // self.n_layers)
            )
        elif audio_context:
            encoder_outputs, encoder_hidden = self.encoder(
                in_text, in_lengths, in_audio, None
            )
        elif use_TCN:
            encoder_outputs, encoder_hidden = self.encoder(in_text)
        else:
            encoder_outputs, encoder_hidden = self.encoder(in_text, in_lengths, None)

        decoder_hidden = encoder_hidden[
            : self.decoder.n_layers
        ]  # use last hidden state from encoder
        if noisy:
            eps = torch.randn_like(decoder_hidden)
            decoder_hidden = eps * 10  # torch.cat([decoder_hidden, eps/10], dim=2)

        # run through decoder one time step at a time
        decoder_input = poses[0]  # initial pose from the dataset
        if self.text2_embedding_discrete:
            outputs[0] = onehottie[0]  # .float()
        else:
            outputs[0] = decoder_input
        # decoder_input = outputs[0]
        dd = decoder_input.detach().cpu().numpy()
        # outputs[0] = onehottie[0]  # decoder_input

        # todo: it could be considered in the pose itself at the inference time
        if vid_indices is not None:  # for the inference
            decoder_input = vid_indices
            decoder_output, decoder_hidden, attentions = self.decoder(
                None, decoder_input, decoder_hidden, encoder_outputs, vid_indices
            )

            outputs[0] = decoder_output
            decoder_input = outputs[0].argmax(1)

        attentions_list = []
        q = self.sentence_frame_length // self.n_frames
        if debug:
            print(
                "self.sentence_frame_length//self.n_frames",
                self.sentence_frame_length // self.n_frames,
            )
        for t in range(1, self.sentence_frame_length // self.n_frames):  # +2):
            if debug:
                print("decoder_input", decoder_input.shape)
            # decoder_input = torch.tensor([10]).to(decoder_input.device)
            decoder_output, decoder_hidden, attentions = self.decoder(
                None, decoder_input, decoder_hidden, encoder_outputs, vid_indices
            )
            # For sake of visualization
            attentions_list.append(attentions)

            if self.text2_embedding_discrete == True:
                if debug:
                    print(self.encoder)
                    print(self.decoder)
                    print("Check discrete:", decoder_output.shape)
                    print(
                        "Check discrete: argmax",
                        decoder_output.argmax(1),
                        decoder_output.argmax(1).shape,
                    )
                # outputs[t] = torch.nn.functional.softmax(decoder_output,dim=1)
                # outputs[t] =\
                # decoder_output = decoder_output.argmax(1)
                # print(q.shape, 'qq')
                # outputs[t] = torch.argmax(torch.nn.functional.softmax(decoder_output, dim=1), 0) #[128]
                # print(outputs[t])
                # print(outputs[t].shape)
                # print("")
            # decoder_output = torch.nn.functional.softmax(decoder_output, dim=1)
            outputs[t] = decoder_output  # Todo: this is the problem int <- float
            # Todo: maybe we need a softmax right here
            # print(outputs.shape)

            if t < self.n_pre_poses:
                decoder_input = poses[
                    t
                ]  # onehottie[t].float() #poses[t]  # next input is current target
            else:
                # 0th dimension is batch size, 1st dimension is word embedding
                best_guess = decoder_output.argmax(1)
                decoder_input = decoder_output  # next input is current prediction

                if self.text2_embedding_discrete:
                    decoder_input = best_guess
        # print(outputs.transpose(0, 1).shape)
        return outputs.transpose(0, 1), attentions_list


# Todo: new _____________
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device='cpu'
class EncoderRNN_New(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layer: int = 2,
        pre_trained_embedding: np.ndarray | None = None,
    ):
        """ """
        super(EncoderRNN_New, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.embed_size = 300
        if (
            pre_trained_embedding is not None
        ):  # use pre-trained embedding (e.g., word2vec, glove)
            assert pre_trained_embedding.shape[0] == input_size
            assert pre_trained_embedding.shape[1] == self.embed_size
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pre_trained_embedding), freeze=False
            )
        else:
            self.embedding = nn.Embedding(input_size, self.embed_size)

        self.gru = nn.GRU(
            self.embed_size, hidden_size, num_layers=self.n_layer, bidirectional=True
        )

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        embedded = self.embedding(input)  # .view(1, 1, -1)
        output = embedded.unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self) -> torch.Tensor:
        """ """
        return torch.zeros(
            self.n_layer * (1 + self.gru.bidirectional),
            128,
            self.hidden_size,
            device=device,
        )


class DecoderRNN_New(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(self, hidden_size: int, output_size: int, n_layer: int = 2):
        """ """
        super(DecoderRNN_New, self).__init__()
        self.hidden_size = hidden_size
        self.ouput_size = output_size
        self.n_layer = n_layer

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layer)
        self.fc_out = nn.Linear(hidden_size, output_size)
        # self.softmax = F.softmax(dim=1)

        # self.pre_linear = nn.Sequential(
        #     nn.Linear(linear_input_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(inplace=True)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        output = self.embedding(input)  # .view(1, 1, -1)
        output = (output).unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        output = self.fc_out(output)
        a = output.cpu().detach().numpy()
        b = a.argmax()
        return output, hidden

    def initHidden(self) -> torch.Tensor:
        """ """
        return torch.zeros(1, 1, self.hidden_size, device=device)


MAX_LENGTH = 4


class AttnDecoderRNN_New(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dropout_p: float = 0.1,
        max_length: int = MAX_LENGTH,
    ):
        """ """
        super(AttnDecoderRNN_New, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        embedded = self.embedding(input)  # .view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self) -> torch.Tensor:
        """ """
        return torch.zeros(1, 1, self.hidden_size, device=device)


class text2embedding_model_New(nn.Module):
    """
    This class documentation is #TODO.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        pose_dim: int,
        n_frames: int,
        n_words: int,
        word_embed_size: int,
        word_embeddings: np.ndarray,
        speaker_model=None,
    ):
        super().__init__()
        self.n_layer = 1
        self.encoder = EncoderRNN_New(
            3863, args.hidden_size, self.n_layer, pre_trained_embedding=word_embeddings
        )

        # self.encoder = EncoderRNN(
        #     n_words, word_embed_size, args.hidden_size, args.n_layers,
        #     dropout=args.dropout_prob, pre_trained_embedding=word_embeddings)

        pose_dim = int(args.autoencoder_vq_components) + 2
        self.decoder = DecoderRNN_New(args.hidden_size, pose_dim, self.n_layer)
        self.max_length = 90
        self.SOS_token = 512
        self.eos_token = 513

    def forward(
        self,
        in_text: torch.Tensor,
        in_lengths: torch.Tensor,
        poses: torch.Tensor,
        vid_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """ """
        # reshape to (seq x batch x dim)
        in_text = in_text.transpose(0, 1)
        poses = poses.transpose(0, 1)

        input_length = in_text.size()[0]
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(
            self.max_length,
            in_text.shape[1],
            self.encoder.hidden_size * (1 + self.encoder.gru.bidirectional),
            device=device,
        )

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(in_text[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output

        # Decoder
        target_length = poses.size(0)
        decoder_outputss = (
            torch.zeros(poses.shape[0], poses.shape[1], self.decoder.ouput_size)
            .float()
            .to(device)
        )
        decoder_input = torch.tensor([self.SOS_token], device=device)
        decoder_input = decoder_input.repeat(poses.shape[1])

        decoder_input = poses[0]
        x = F.one_hot(poses[0], 514)
        q = decoder_outputss[0]
        decoder_outputss[0] = F.one_hot(poses[0], 514)
        decoder_hidden = (
            encoder_hidden[: self.decoder.n_layer]
            + encoder_hidden[self.decoder.n_layer :]
        )

        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                # decoder_output, decoder_hidden, decoder_attention = self.decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                # loss += criterion(decoder_output, target_tensor[di])
                decoder_input = poses[di]  # Teacher forcing
                decoder_outputss[di] = decoder_output
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(1, target_length):
                # decoder_output, decoder_hidden, decoder_attention = self.decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoder_outputss[di] = decoder_output

        return decoder_outputss


# Todo: Audio ---------------
