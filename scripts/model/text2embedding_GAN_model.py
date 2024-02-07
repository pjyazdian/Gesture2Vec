import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Based on the following Se2Seq implementations:
- https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
- https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
'''
debug = False

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout

        if pre_trained_embedding is not None:  # use pre-trained embedding (e.g., word2vec, glove)
            assert pre_trained_embedding.shape[0] == input_size
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=False)
        else:
            self.embedding = nn.Embedding(input_size, embed_size)

        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        :param input_seqs:
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input_lengths:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.1,
                 discrete_representation=False, speaker_model=None):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size
        if noisy:
            self.hidden_size = hidden_size*2
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.discrete_representation = discrete_representation
        self.speaker_model = speaker_model

        # define embedding layer
        if self.discrete_representation:
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.dropout = nn.Dropout(dropout_p)

        if self.speaker_model:
            self.speaker_embedding = nn.Embedding(speaker_model.n_words, 8)

        # calc input size
        if self.discrete_representation:
            input_size = hidden_size  # embedding size
        linear_input_size = input_size + hidden_size
        if self.speaker_model:
            linear_input_size += 8

        # define layers
        self.pre_linear = nn.Sequential(
            nn.Linear(linear_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)

        # self.out = nn.Linear(hidden_size * 2, output_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def freeze_attn(self):
        for param in self.attn.parameters():
            param.requires_grad = False

    def forward(self, motion_input, last_hidden, encoder_outputs, vid_indices=None):
        '''
        :param motion_input:
            motion input for current time step, in shape [batch x dim]
        :param last_hidden:
            last hidden state of the decoder, in shape [layers x batch x hidden_size]
        :param encoder_outputs:
            encoder outputs in shape [steps x batch x hidden_size]
        :param vid_indices:
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        '''

        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        if self.discrete_representation:
            word_embedded = self.embedding(motion_input).view(1, motion_input.size(0), -1)  # [1 x B x embedding_dim]
            motion_input = self.dropout(word_embedded)
        else:
            motion_input = motion_input.view(1, motion_input.size(0), -1)  # [1 x batch x dim]

        # attention
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)  # [batch x 1 x T]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [batch x 1 x attn_size]
        context = context.transpose(0, 1)  # [1 x batch x attn_size]

        # make input vec
        rnn_input = torch.cat((motion_input, context), 2)  # [1 x batch x (dim + attn_size)]

        if self.speaker_model:
            assert vid_indices is not None
            speaker_context = self.speaker_embedding(vid_indices).unsqueeze(0)
            rnn_input = torch.cat((rnn_input, speaker_context), 2)  # [1 x batch x (dim + attn_size + embed_size)]

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

        return output, hidden, attn_weights


class Generator(nn.Module):
    def __init__(self, args, motion_dim, discrete_representation=False, speaker_model=None):
        super(Generator, self).__init__()
        self.output_size = motion_dim
        self.n_layers = args.n_layers
        self.discrete_representation = discrete_representation
        self.decoder = BahdanauAttnDecoderRNN(input_size=motion_dim,
                                              hidden_size=args.hidden_size,
                                              output_size=self.output_size,
                                              n_layers=self.n_layers,
                                              dropout_p=args.dropout_prob,
                                              discrete_representation=discrete_representation,
                                              speaker_model=speaker_model)

    def freeze_attn(self):
        self.decoder.freeze_attn()

    def forward(self, z, motion_input, last_hidden, encoder_output, vid_indices=None):
        if z is None:
            input_with_noise_vec = motion_input
        else:
            assert not self.discrete_representation  # not valid for discrete representation
            input_with_noise_vec = torch.cat([motion_input, z], dim=1)  # [bs x (10+z_size)]

        return self.decoder(input_with_noise_vec, last_hidden, encoder_output, vid_indices)

noisy = False
class text2embedding_model(nn.Module):
    def __init__(self, args, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, speaker_model=None):
        super().__init__()

        # Todo: find a better way to play with representation learning dimension
        # Todo: Regarding hidden size, sentence time length, etc.
        pose_dim = args.n_layers * args.hidden_size

        self.encoder = EncoderRNN(
            n_words, word_embed_size, args.hidden_size, args.n_layers,
            dropout=args.dropout_prob, pre_trained_embedding=word_embeddings)
        self.decoder = Generator(args, pose_dim, speaker_model=speaker_model)

        self.n_frames = n_frames
        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = pose_dim
        self.sentence_frame_length = args.sentence_frame_length

    def forward(self, in_text, in_lengths, poses, vid_indices):
        # reshape to (seq x batch x dim)
        in_text = in_text.transpose(0, 1)
        poses = poses.transpose(0, 1)


        outputs = torch.zeros(self.sentence_frame_length//self.n_frames, poses.size(1),
                              self.decoder.output_size)\
            .to(poses.device)
        if debug:
            print("outputs", outputs.shape)



        # run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(in_text, in_lengths, None)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder
        if noisy:
            eps = torch.randn_like(decoder_hidden)
            decoder_hidden = eps*10 # torch.cat([decoder_hidden, eps/10], dim=2)

        # run through decoder one time step at a time
        # decoder_input = poses[0]  # initial pose from the dataset
        decoder_input = torch.zeros(( poses.size(1), self.decoder.output_size) ).to(poses.device)
        outputs[0] = decoder_input

        q = self.sentence_frame_length//self.n_frames
        if debug:
            print("self.sentence_frame_length//self.n_frames", self.sentence_frame_length//self.n_frames)
        for t in range(0, self.sentence_frame_length//self.n_frames):
            decoder_output, decoder_hidden, _ = self.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                             vid_indices)
            outputs[t] = decoder_output

            # if t < self.n_pre_poses:
            #     decoder_input = poses[t]  # next input is current target
            # else:
            decoder_input = decoder_output  # next input is current prediction

        return outputs.transpose(0, 1)


class Generator_gan(nn.Module):

    def __init__(self, i_size, n_size, o_size, h_size):
        super(Generator, self).__init__()

        self.i_size = i_size
        self.n_size = n_size
        self.o_size = o_size

        self.i_fc = nn.Linear(i_size, int(h_size / 2))
        self.n_fc = nn.Linear(n_size, int(h_size / 2))
        self.o_fc = nn.Linear(2 * h_size, o_size)
        self.rnn = nn.LSTM(h_size, h_size, num_layers=2, bidirectional=True)

    def forward(self, i, n):
        """3D tensor"""

        assert len(
            i.shape) == 3, f"expect 3D tensor with shape (t, n, dim), got {i.shape}"
        assert n.size(
            0) == 1, f"shape of noise must be (1, N, dim), got noise with {n.size(0)}"
        assert i.size(1) == n.size(
            1), f"batch size of input and noise must be the same, got input with {i.size(1)}, noise with {n.size(1)}"

        n = n.repeat(i.size(0), 1, 1)
        t, bs = i.size(0), i.size(1)
        i = i.view(t * bs, -1)
        n = n.view(t * bs, -1)
        i = F.leaky_relu(self.i_fc(i), 1e-2)
        n = F.leaky_relu(self.n_fc(n), 1e-2)
        i = i.view(t, bs, -1)
        n = n.view(t, bs, -1)
        x = torch.cat([i, n], dim=-1)
        x, _ = self.rnn(x)
        x = F.leaky_relu(x, 1e-2)
        o = self.o_fc(x)
        return o

    def forward_given_noise_seq(self, i, n):
        t, bs = i.size(0), i.size(1)
        i = i.view(t * bs, -1)
        n = n.view(t * bs, -1)
        i = F.leaky_relu(self.i_fc(i), 1e-2)
        n = F.leaky_relu(self.n_fc(n), 1e-2)
        i = i.view(t, bs, -1)
        n = n.view(t, bs, -1)
        x = torch.cat([i, n], dim=-1)
        x, _ = self.rnn(x)
        x = F.leaky_relu(x, 1e-2)
        o = self.o_fc(x)
        return o

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class Discriminator_gan(nn.Module):

    def __init__(self, i_size, c_size, h_size):
        super(Discriminator_gan, self).__init__()

        self.i_size = i_size
        self.c_szie = c_size

        self.i_fc = nn.Linear(i_size, int(h_size / 2))
        self.c_fc = nn.Linear(c_size, int(h_size / 2))
        self.rnn = nn.LSTM(h_size, h_size, num_layers=2, bidirectional=True)
        self.o_fc = nn.Linear(2 * h_size, 1)

    def forward(self, x, c):
        """3D tensor + 3D tensor"""

        assert len(x.shape) == 3, f"expect 3D tensor, got {x.shape}"

        t, bs = x.size(0), x.size(1)
        x = x.view(t * bs, -1)
        c = c.view(t * bs, -1)
        x = F.leaky_relu(self.i_fc(x))
        c = F.leaky_relu(self.c_fc(c))
        x = x.view(t, bs, -1)
        c = c.view(t, bs, -1)
        x = torch.cat([x, c], dim=-1)
        x, _ = self.rnn(x)
        x = x.view(t * bs, -1)
        x = torch.sigmoid(self.o_fc(x))
        x = x.view(t, bs, -1)
        return x

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class EncoderRNN_discriminator(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout

        if pre_trained_embedding is not None:  # use pre-trained embedding (e.g., word2vec, glove)
            assert pre_trained_embedding.shape[0] == input_size
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=False)
        else:
            self.embedding = nn.Embedding(input_size, embed_size)

        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        self.f_discriminate = nn.Linear(n_layers * hidden_size, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        :param input_seqs:
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input_lengths:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class text2embedding_model_generator(nn.Module):
    def __init__(self, args, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, speaker_model=None):
        super().__init__()

        # Todo: find a better way to play with representation learning dimension
        # Todo: Regarding hidden size, sentence time length, etc.

        pose_dim = args.n_layers * args.hidden_size
        pose_dim = 400
        self.encoder_text = EncoderRNN(
            n_words, word_embed_size, args.hidden_size // 2, args.n_layers,
            dropout=args.dropout_prob, pre_trained_embedding=word_embeddings)
        self.decoder = Generator(args, pose_dim, speaker_model=speaker_model)

        self.fc_input = nn.Linear(args.hidden_size // 2, args.hidden_size // 2)
        self.fc_noise = nn.Linear(args.noise_dim, args.hidden_size // 2)

        self.n_frames = n_frames
        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = pose_dim
        self.sentence_frame_length = args.sentence_frame_length
        self.noise_dim = args.noise_dim
    def forward(self, in_text, in_lengths, poses, vid_indices, ):
        # reshape to (seq x batch x dim)
        in_text = in_text.transpose(0, 1)
        poses = poses.transpose(0, 1)

        outputs = torch.zeros(self.sentence_frame_length // self.n_frames, poses.size(1),
                              self.decoder.output_size) \
            .to(poses.device)
        if debug:
            print("outputs", outputs.shape)

        # run words through encoder
        encoder_outputs, encoder_hidden = self.encoder_text(in_text, in_lengths, None)
        if debug:
            print("encoder_hidden from lstm", encoder_hidden.shape)
            print("Encoder_text", self.encoder_text)
        encoder_hidden = F.leaky_relu(self.fc_input(encoder_hidden), 1e-2)

        noise = np.random.normal(0, 1, (encoder_hidden.shape[0]* encoder_hidden.shape[1], self.noise_dim))
        noise = torch.from_numpy(noise).float().to(encoder_hidden.device)
        if debug:
            print("Zarb", encoder_hidden.shape[0]* encoder_hidden.shape[1])
            print("noise.shape1", noise.shape)
            print("encoder_hidden.shape1", encoder_hidden.shape)
        noise = F.leaky_relu(self.fc_noise(noise), 1e-2)
        if debug:
            print("after leaky", noise.shape)
        noise = noise.reshape(encoder_hidden.shape)
        if debug:
            print("noise2", noise.shape)
            print("encoder_hidden2", encoder_hidden.shape)

        decoder_hidden = torch.cat([encoder_hidden[:self.decoder.n_layers],
                                    noise[:self.decoder.n_layers]], dim=-1)

        encoder_outputs = torch.cat([encoder_outputs[:self.decoder.n_layers],
                                    noise[:self.decoder.n_layers]], dim=-1)
        if debug:
            print(" cat !decoder_hidden", decoder_hidden.shape)


        # decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder
        if noisy:
            eps = torch.randn_like(decoder_hidden)
            decoder_hidden = eps * 10  # torch.cat([decoder_hidden, eps/10], dim=2)

        # run through decoder one time step at a time
        # decoder_input = poses[0]  # initial pose from the dataset
        decoder_input = torch.zeros((poses.size(1), self.decoder.output_size)).to(poses.device)
        outputs[0] = decoder_input

        q = self.sentence_frame_length // self.n_frames
        if debug:
            print("self.sentence_frame_length//self.n_frames", self.sentence_frame_length // self.n_frames)
        for t in range(0, self.sentence_frame_length // self.n_frames):
            decoder_output, decoder_hidden, _ = self.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                             vid_indices)
            outputs[t] = decoder_output

            # if t < self.n_pre_poses:
            #     decoder_input = poses[t]  # next input is current target
            # else:
            decoder_input = decoder_output  # next input is current prediction

        return outputs.transpose(0, 1)


class text2embedding_model_discriminator(nn.Module):
    def __init__(self, args, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, speaker_model=None):
        super().__init__()

        # Todo: find a better way to play with representation learning dimension
        # Todo: Regarding hidden size, sentence time length, etc.
        pose_dim = args.n_layers * args.hidden_size

        self.encoder_text = EncoderRNN(
            n_words, word_embed_size, args.hidden_size // 2, args.n_layers,
            dropout=args.dropout_prob, pre_trained_embedding=word_embeddings)
        # self.decoder = Generator(args, pose_dim, speaker_model=speaker_model)

        # Todo: fix this 400 dim
        self.encoder_pose_latent = nn.GRU(400, args.hidden_size // 2, args.n_layers,
                                          dropout=args.dropout_prob, bidirectional=True)

        self.decoder = nn.GRU(args.hidden_size, args.hidden_size, args.n_layers,
                              dropout=args.dropout_prob, bidirectional=True)

        self.fc_text = nn.Linear(args.hidden_size // 2, args.hidden_size // 2)
        self.fc_pose = nn.Linear(args.hidden_size // 2, args.hidden_size // 2)
        self.classifier = nn.Linear(args.hidden_size * args.n_layers, 1)

        self.n_frames = n_frames
        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = pose_dim
        self.sentence_frame_length = args.sentence_frame_length
        self.n_layers = args.n_layers
    def forward(self, in_text, in_lengths, poses, vid_indices):
        # reshape to (seq x batch x dim)
        in_text = in_text.transpose(0, 1)
        poses = poses.transpose(0, 1)

        # outputs = torch.zeros(self.sentence_frame_length // self.n_frames, poses.size(1),
        #                       self.decoder.output_size) \
        #     .to(poses.device)
        # if debug:
        #     print("outputs", outputs.shape)

        # run words through encoder
        # 1. Encoders
        encoder_text_outputs, encoder_text_hidden = self.encoder_text(in_text, in_lengths, None)
        encoder_pose_outputs, encoder_pose_hidden = self.encoder_pose_latent(poses, None)
        encoder_text_hidden = torch.nn.functional.leaky_relu(self.fc_text(encoder_text_hidden), 1e-2)
        encoder_pose_hidden = torch.nn.functional.leaky_relu(self.fc_pose(encoder_pose_hidden), 1e-2)

        if debug:
            print("encoder_text_outputs", encoder_text_outputs.shape)
            print("encoder_pose_outputs", encoder_pose_outputs.shape)
        # 2. Decoder
        decoder_hidden = torch.cat([encoder_text_hidden[:self.n_layers],
                                    encoder_pose_hidden[:self.n_layers]], dim=-1)


        # decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder
        if noisy:
            eps = torch.randn_like(decoder_hidden)
            decoder_hidden = eps * 10  # torch.cat([decoder_hidden, eps/10], dim=2)


        # Todo: should find a way to make surw that if it's ok to use pose encoder output, not text or combined
        # decoder_input = torch.zeros_like(poses)
        decoder_input = encoder_pose_outputs
        if self.decoder.bidirectional:
            decoder_hidden = torch.cat([decoder_hidden, decoder_hidden], dim=0)
        if debug:
            print("D: decoder_input", decoder_input.shape)
            print("D decoder_hidden", decoder_hidden.shape)
            print(self.decoder)

        decoder_outputs, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        # run through decoder one time step at a time
        # decoder_input = poses[0]  # initial pose from the dataset
        # decoder_input = torch.zeros(( poses.size(1), self.decoder.output_size) ).to(poses.device)
        # outputs[0] = decoder_input

        if debug:
            print("Final decoder hidden", decoder_hidden.shape)
            print("Final calssifier", self.classifier)
        decoder_hidden = decoder_hidden[:self.n_layers]
        decoder_hidden = decoder_hidden.transpose(1,0)
        decoder_hidden = decoder_hidden.reshape(decoder_hidden.shape[0], -1)
        if debug:
            print("Final decoder hidden", decoder_hidden.shape)

        test = (self.classifier(decoder_hidden))
        test = test.squeeze()
        if debug:
            print("test squeeze", test.shape)
        test =torch.sigmoid(test)
        if debug:
            print("test sigmoid", test.shape)
        output = torch.sigmoid(self.classifier(decoder_hidden.squeeze()))
        if debug:
            print("TEsT:,", test.shape)
            print("Discriminator output", output.shape)
        return output
        # return outputs.transpose(0, 1)









class text2embedding_model_gan(nn.Module):
    def __init__(self, args, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, speaker_model=None):
        super().__init__()

        self.args = args
        self.generator = text2embedding_model_generator(args, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, speaker_model=None)

        self.discriminator = text2embedding_model_discriminator(args, pose_dim, n_frames, n_words, word_embed_size,
                                                        word_embeddings, speaker_model=None)
        print("Model initiated!")
    def forward(self, in_text, in_lengths, poses, vid_indices):
       return None
