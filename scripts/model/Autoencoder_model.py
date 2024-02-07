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
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.in_layer = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, input_seqs, hidden=None):
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
        if debug:
            print("Encoder_Forward_input", input_seqs.shape)
        input_seq_in_layered = self.in_layer(input_seqs)
        outputs, hidden = self.gru(input_seq_in_layered, hidden)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        if debug:
            print("Encoder_Forward_ out, hidden", outputs.shape, hidden.shape)
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
    def __init__(self, args, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.1,
                 discrete_representation=False, speaker_model=None):
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

        # if self.speaker_model:
        #     self.speaker_embedding = nn.Embedding(speaker_model.n_words, 8)

        # calc input size
        # if self.discrete_representation:
        #     input_size = hidden_size  # embedding size
        # linear_input_size = input_size + hidden_size
        # if self.speaker_model:
        #     linear_input_size += 8

        if args.autoencoder_conditioned == 'True':
            self.autoencoder_conditioned = True
        else:
            self.autoencoder_conditioned = False

        # define layers

        if args.autoencoder_att == 'True':
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
            nn.ReLU(inplace=True)
        )


        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        if args.autoencoder_fixed_weight == 'True':
            self.autoencoder_fixed_weight = True
            for param in self.gru.parameters():
                param.requires_grad = False



        # self.out = nn.Linear(hidden_size * 2, output_size)
        self.out_layer = nn.Linear(hidden_size, output_size)

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

        if debug:
            print("Decoder_forward, motion_input", motion_input.shape) # [1, 128, 41])
        if debug:
            print("Decoder_forward, encoder_outputs", encoder_outputs.shape)  # [1, 128, 41])

        # attention
        if self.att_use:
            attn_weights = self.attn(last_hidden[-1], encoder_outputs)  # [batch x 1 x T]
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [batch x 1 x attn_size]
            context = context.transpose(0, 1)  # [1 x batch x attn_size]

            # make input vec
            rnn_input = torch.cat((motion_input, context), 2)  # [1 x batch x (dim + attn_size)]
        else:
            attn_weights = None
            # rnn_input = torch.cat((motion_input, encoder_outputs), 2)  # [1 x batch x (dim + attn_size)]
            rnn_input = motion_input
        if debug:
            print("Decoder_forward, rnn_input", rnn_input.shape) # [1, 128, 241])

        # Check if unconditioned
        # Todo: Oh my god
        # print(self.autoencoder_conditioned)
        if self.autoencoder_conditioned==False:
            rnn_input = torch.zeros_like(rnn_input)

        # if self.speaker_model:
        #     assert vid_indices is not None
        #     speaker_context = self.speaker_embedding(vid_indices).unsqueeze(0)
        #     rnn_input = torch.cat((rnn_input, speaker_context), 2)  # [1 x batch x (dim + attn_size + embed_size)]

        q = rnn_input.squeeze(0)
        rnn_input = self.pre_linear(rnn_input.squeeze(0))
        if debug:
            print("Decoder_forward, rnn_input", rnn_input.shape) # torch.Size([128, 200])

        rnn_input = rnn_input.unsqueeze(0)
        if debug:
            print("Decoder_forward, rnn_input", rnn_input.shape) # torch.Size([1, 128, 200])

        # rnn

        output, hidden = self.gru(rnn_input, last_hidden)

        # post-fc
        output = output.squeeze(0)  # [1 x batch x hidden_size] -> [batch x hidden_size]
        output = self.out_layer(output)

        return output, hidden, attn_weights


class Generator(nn.Module):
    def __init__(self, args, motion_dim, discrete_representation=False, speaker_model=None):
        super(Generator, self).__init__()
        self.output_size = motion_dim
        self.n_layers = args.n_layers
        self.discrete_representation = discrete_representation
        self.decoder = BahdanauAttnDecoderRNN(args=args, input_size=args.rep_learning_dim,
                                              hidden_size=args.hidden_size,
                                              output_size=args.rep_learning_dim,
                                              n_layers=self.n_layers,
                                              dropout_p=args.dropout_prob,
                                              discrete_representation=discrete_representation,
                                              speaker_model=speaker_model)
        self.is_training = True
    def freeze_attn(self):
        self.decoder.freeze_attn()

    def forward(self, z, motion_input, last_hidden, encoder_output, vid_indices=None):
        if z is None:
            input_with_noise_vec = motion_input
        else:
            assert not self.discrete_representation  # not valid for discrete representation
            input_with_noise_vec = torch.cat([motion_input, z], dim=1)  # [bs x (10+z_size)]

        return self.decoder(input_with_noise_vec, last_hidden, encoder_output, vid_indices)


class Autoencoder_seq2seq(nn.Module):
    def __init__(self, args, pose_dim, n_frames):
        super().__init__()
        self.encoder = EncoderRNN(
            args.rep_learning_dim, args.hidden_size, args.n_layers,
            dropout=args.dropout_prob, pre_trained_embedding=None)
        self.decoder = Generator(args, pose_dim, speaker_model=None)

        self.n_frames = n_frames
        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = args.rep_learning_dim

        if args.autoencoder_conditioned == 'True':
            self.autoencoder_conditioned = True
        else:
            self.autoencoder_conditioned = False

        try:
            if args.autoencoder_vae == 'True':
                self.VAE = True
                self.VAE_fc_mean = nn.Linear(self.decoder.n_layers*args.hidden_size,
                                             self.decoder.n_layers*args.hidden_size)
                self.VAE_fc_std = nn.Linear(self.decoder.n_layers*args.hidden_size,
                                            self.decoder.n_layers*args.hidden_size)
                self.VAE_fc_decoder = nn.Linear(self.decoder.n_layers*args.hidden_size,
                                                self.decoder.n_layers*args.hidden_size)
            else:
                self.VAE = False
        except:
            self.VAE = False

        self.do = nn.Dropout(args.dropout_prob)

        # Todo: should fix, currently we freez from train_epoch function
        # if args.autoencoder_freeze_encoder=='True':
        #     self.freez_encoder()

    def freez_encoder(self):
        self.autoencoder_fixed_encoder_weight = True

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.VAE_fc_mean.parameters():
            param.requires_grad = False
        for param in self.VAE_fc_std.parameters():
            param.requires_grad = False
        for param in self.VAE_fc_decoder.parameters():
            param.requires_grad = False



    def reparameterize(self, mu, logVar, train=True):

        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        # Todo: fix this for train and test time checking
        if train:
            return mu + std * eps
        else:
            return mu # + std * eps

    def forward(self, in_poses, out_poses):
        # reshape to (seq x batch x dim)
        # in_text = in_text.transpose(0, 1)
        in_poses = in_poses.transpose(0, 1)
        in_poses = self.do(in_poses)
        out_poses = out_poses.transpose(0, 1)

        outputs = torch.zeros(self.n_frames, out_poses.size(1), self.decoder.output_size).to(out_poses.device)

        # run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(in_poses, None)



        # print("Sanity check:", encoder_hidden[:self.decoder.n_layers].shape)
        # self.VAE = False
        if self.VAE:
            if debug:
                print("self.VAE", self.VAE)
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder
            # [2, 128, 200]
            # print("decoder_hidden!!! org", decoder_hidden.shape)
            decoder_hidden = decoder_hidden.transpose(1, 0).contiguous() # [128, 2, 200]
            decoder_hidden = torch.reshape(decoder_hidden, (decoder_hidden.shape[0], -1))
            mean = self.VAE_fc_mean(decoder_hidden)
            logvar = self.VAE_fc_std(decoder_hidden)
            z = self.reparameterize(mean, logvar)
            z = self.VAE_fc_decoder(z)
            decoder_hidden = z.reshape(decoder_hidden.shape[0],
                                       self.decoder.n_layers, -1)
            decoder_hidden = decoder_hidden.transpose(1 ,0).contiguous()
            # print("decoder_hidden!!! modified", decoder_hidden.shape)
            decoder_first_hidden = decoder_hidden
        else:
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder
            # print("decoder_hidden!!! not VAE ", decoder_hidden.shape)

        decoder_first_hidden = decoder_hidden
        # run through decoder one time step at a time
        decoder_input = out_poses[0]  # initial pose from the dataset
        outputs[0] = decoder_input

        # Todo: should be fixed such a parameter for inference time
        all_hiddens = []



        for t in range(1, self.n_frames):
            decoder_output, decoder_hidden, _ = self.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                             None)
            outputs[t] = decoder_output

            if t < self.n_pre_poses:
                decoder_input = out_poses[t]  # next input is current target
            else:
                decoder_input = decoder_output  # next input is current prediction
            if not self.autoencoder_conditioned:
                decoder_input = torch.zeros_like(decoder_input)

        if self.VAE:
            return outputs.transpose(0, 1),\
                   decoder_first_hidden[:self.decoder.n_layers],\
                   mean, logvar
        else:
            return outputs.transpose(0, 1),\
                   decoder_first_hidden[:self.decoder.n_layers]
