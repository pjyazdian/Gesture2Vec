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

class EncoderCNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):
        super(EncoderCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.in_layer = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        self.cnn = nn.Conv1d(in_channels=1,out_channels=2,kernel_size=2, stride=2)
        channels = [hidden_size, hidden_size, hidden_size, hidden_size, hidden_size, hidden_size]
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels[0],out_channels=channels[1],kernel_size=5, stride=5, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channels[1],out_channels=channels[2],kernel_size=3, stride=3, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channels[2],out_channels=channels[3],kernel_size=2, stride=2, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
            # nn.Conv1d(in_channels=channels[3],out_channels=channels[4],kernel_size=2, stride=2, bias=False),
            # nn.BatchNorm1d(channels[4], affine=True),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(in_channels=channels[4],out_channels=channels[5],kernel_size=2, stride=2, bias=False),
        )



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
        input_seq_in_layered = input_seq_in_layered.permute((1,2,0)) #128,30,200 --> 128,200,30 n*c*l
        # outputs, hidden = self.gru(input_seq_in_layered, hidden)


        print(self.cnn)
        hidden = self.cnn(input_seq_in_layered)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        outputs = torch.tensor(0)
        if debug:
            print("Encoder_Forward_ out, hidden", outputs.shape, hidden.shape)
        return outputs, hidden
class DecoderCNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):
        super(DecoderCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.in_layer = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        self.cnn = nn.Conv1d(in_channels=1,out_channels=2,kernel_size=2, stride=2)
        channels = [hidden_size, hidden_size, hidden_size, hidden_size, hidden_size, hidden_size]
        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(in_channels=channels[0],out_channels=channels[1],kernel_size=2, stride=2, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(in_channels=channels[1],out_channels=channels[2],kernel_size=3, stride=3, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(in_channels=channels[2],out_channels=channels[3],kernel_size=5, stride=5, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),

            # nn.Conv1d(in_channels=channels[3],out_channels=channels[4],kernel_size=2, stride=2, bias=False),
            # nn.BatchNorm1d(channels[4], affine=True),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(in_channels=channels[4],out_channels=channels[5],kernel_size=2, stride=2, bias=False),
        )



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
            print(self.in_layer)
        input_seq_in_layered = self.in_layer(input_seqs.squeeze()).unsqueeze(-1)
        input_seq_in_layered = input_seq_in_layered.unsqueeze(0)
        # input_seq_in_layered = input_seq_in_layered.permute((1,2,0)) #128,30,200 --> 128,200,30 n*c*l
        # outputs, hidden = self.gru(input_seq_in_layered, hidden)

        print("DCNN input_seq_in_layered", input_seq_in_layered.shape)

        outputs = self.cnn(input_seq_in_layered)
        # print("DCNN outputs", outputs.shape)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        hidden = torch.tensor(0)
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
        if self.autoencoder_conditioned==False:
            rnn_input = torch.zeros_like(rnn_input)
        rnn_input = nn.Dropout(0.95)(rnn_input)

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


class Autoencoder_VQVAE(nn.Module):
    def __init__(self, args, pose_dim, n_frames):
        super().__init__()

        self.CNN = False


        # 1. Defining Encoder and Decoder
        self.encoder = EncoderRNN(
            args.rep_learning_dim, args.hidden_size, args.n_layers,
            dropout=args.dropout_prob, pre_trained_embedding=None)

        self.out_layer_encoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.Tanh())
        self.out_layer_decoder = nn.Sequential(
            nn.Linear(args.hidden_size, pose_dim),
            )

        self.decoder = Generator(args, pose_dim, speaker_model=None)

        if(self.CNN):
            self.encoder = EncoderCNN(
                args.rep_learning_dim, args.hidden_size, args.n_layers,
                dropout=args.dropout_prob, pre_trained_embedding=None)
            self.decoder = DecoderCNN(
                args.hidden_size, args.hidden_size, args.n_layers,
                dropout=args.dropout_prob, pre_trained_embedding=None)



        # Defining Variational Autoencoder layers
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
        

        # 2. Defining  Vector Quantizer layer
        if args.autoencoder_vq == 'True':
            decay = 0.85
            self.vq = True
            self.vq_components = int(args.autoencoder_vq_components)
            self.commitment_cost = float(args.autoencoder_vq_commitment_cost)
            # Todo: add a pre linear layer before feeding to the VQ layer

            if decay > 0:
                self.vq_layer = VQ_Payam_EMA(self.vq_components,
                                                  args.hidden_size*args.n_layers, self.commitment_cost,
                                                  decay)

            else:
                self.vq_layer = VQ_Payam(self.vq_components,
                                                args.hidden_size*args.n_layers, self.commitment_cost)

            self.vq_layer = VQ_Payam_GSSoft16(self.vq_components,
                                         args.hidden_size * args.n_layers, self.commitment_cost)

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

        if args.autoencoder_conditioned == 'True':
            self.autoencoder_conditioned = True
        else:
            self.autoencoder_conditioned = False




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

    def freeze_VQminuse(self):
        if self.VAE:
            self.freez_encoder()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        for param in self.decoder.parameters():
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

    def forward(self, in_poses, out_poses, vq_layer_active=False):


        # if self.vq and vq_layer_active:
        #     self.freeze_VQminuse()


        # reshape to (seq x batch x dim)
        # in_text = in_text.transpose(0, 1)
        in_poses = in_poses.transpose(0, 1)
        in_poses = self.do(in_poses)
        out_poses = out_poses.transpose(0, 1)

        if not self.CNN:
            outputs = torch.zeros(self.n_frames, out_poses.size(1), self.decoder.output_size).to(out_poses.device)

        # run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(in_poses, None)

        if self.CNN:
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = (encoder_hidden[:self.decoder.n_layers])  # use last hidden state from encoder
        # Todo: I put vector quantization here. However, we need to check it using both VAE and not VAE.
        vq_layer_active = True
        if self.vq and vq_layer_active:
            loss_vq, quantized, perplexity_vq, encodings = self.vq_layer(decoder_hidden)
            decoder_hidden = quantized
        else:
            loss_vq, quantized, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                           torch.tensor(0), torch.tensor(0)

        # print("Sanity check:", encoder_hidden[:self.decoder.n_layers].shape)
        # self.VAE = False
        if self.VAE:
            if debug:
                print("self.VAE", self.VAE)
            # decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # use last hidden state from encoder
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
            outputs = outputs.transpose(1,2)
            # print(outputs.shape)
            shape = outputs.shape
            outputs = outputs.reshape(shape[0]*shape[1], shape[2])
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
                decoder_output, decoder_hidden, _ = self.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                                 None)
                outputs[t] = decoder_output

                if t < self.n_pre_poses:
                    decoder_input = out_poses[t]  # next input is current target
                else:
                    decoder_input = decoder_output  # next input is current prediction
                if not self.autoencoder_conditioned:
                    decoder_input = torch.zeros_like(decoder_input)

        if self.vq:
            if self.VAE:
                return outputs.transpose(0, 1), decoder_first_hidden[:self.decoder.n_layers], mean, logvar, loss_vq, perplexity_vq
            else:
                return outputs.transpose(0, 1), decoder_first_hidden[:self.decoder.n_layers], loss_vq, perplexity_vq
        else:
            if self.VAE:
                return outputs.transpose(0, 1), decoder_first_hidden[:self.decoder.n_layers], mean, logvar
            else:
                return outputs.transpose(0, 1), decoder_first_hidden[:self.decoder.n_layers]


class VQ_Payam(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQ_Payam, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_linear = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._embedding.weight.data.uniform_(-0.2, 0.2 )
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        # self.embedding_grad(False)

    def embedding_grad(self, what):
        for param in self._embedding.parameters():
            param.requires_grad = what

    def forward(self, inputs):

        loss_vq, loss, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                  torch.tensor(0), torch.tensor(0)
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        # flat_input = self.pre_linear(flat_input)


        #__________________________
        # 2. Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # 3. Find nearest encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # print("Idices: ", encoding_indices)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 4. Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight) #.view(input_shape)

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
        #___________________________



        quantized = torch.reshape(flat_input, inputs.shape).contiguous()


        return loss, quantized, perplexity_vq, encodings

class VQ_Payam_EMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
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
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon


    def forward(self, inputs):


        loss_vq, loss, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                  torch.tensor(0), torch.tensor(0)
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        flat_input = self.pre_linear(flat_input)


        #__________________________
        # 2. Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # 3. Find nearest encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # print("Idices: ", encoding_indices)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Todo:
        # Test modification for soft vq
        # ...

        # 4. Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight) #.view(input_shape)
        quantized = torch.reshape(quantized, inputs.shape).contiguous()

        # 5.1 Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))


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
        #___________________________



        quantized = torch.reshape(flat_input, inputs.shape).contiguous()


        return loss, quantized, perplexity_vq, encodings



class VQ_Payam_GSSoft(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQ_Payam_GSSoft, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_linear = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        # self._embedding.weight.data.uniform_(-0.2, 0.2)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        # self.embedding_grad(False)

        self.mean_layer = nn.Linear(embedding_dim, embedding_dim)
        self.logvar_layer = nn.Linear(embedding_dim, num_embeddings)


    def embedding_grad(self, what):
        for param in self._embedding.parameters():
            param.requires_grad = what

    def reparameterize(self, mu, logVar, train=True):

        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        # Todo: fix this for train and test time checking
        if train:
            return mu + std * eps
        else:
            return mu # + std * eps
    def soft_prob(self, dist, smooth):
        dist = (dist)/400
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

    def forward(self, inputs):

        loss_vq, loss, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                  torch.tensor(0), torch.tensor(0)
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        # flat_input = self.pre_linear(flat_input)

        flat_input = self.mean_layer(flat_input)
        z_logvar = self.logvar_layer(flat_input)

        # __________________________
        # 2. Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # # 3. Find nearest encoding
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # # print("Idices: ", encoding_indices)
        # encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # encodings.scatter_(1, encoding_indices, 1)

        smooth = 1. / torch.exp(z_logvar) ** 2
        probs = self.soft_prob(distances, smooth)
        encodings = probs
        # print("---1--", probs.shape, probs)

        # 4. Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)  # .view(input_shape)


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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQ_Payam_GSSoft16, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.reduced_dim = 16
        self.pre_linear = nn.Linear(self._embedding_dim, self.reduced_dim)
        self.post_linear = nn.Linear(self.reduced_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self.reduced_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        # self._embedding.weight.data.uniform_(-0.2, 0.2)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        # self.embedding_grad(False)

        self.mean_layer = nn.Linear(self.reduced_dim, self.reduced_dim)
        self.logvar_layer = nn.Linear(self.reduced_dim, num_embeddings)


    def embedding_grad(self, what):
        for param in self._embedding.parameters():
            param.requires_grad = what

    def reparameterize(self, mu, logVar, train=True):

        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        # Todo: fix this for train and test time checking
        if train:
            return mu + std * eps
        else:
            return mu # + std * eps
    def soft_prob(self, dist, smooth):
        dist = (dist)/400
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

    def forward(self, inputs):

        loss_vq, loss, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                  torch.tensor(0), torch.tensor(0)
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        flat_input = self.pre_linear(flat_input)

        flat_input = self.mean_layer(flat_input)
        z_logvar = self.logvar_layer(flat_input)

        # __________________________
        # 2. Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # # 3. Find nearest encoding
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # # print("Idices: ", encoding_indices)
        # encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # encodings.scatter_(1, encoding_indices, 1)

        smooth = 1. / torch.exp(z_logvar) ** 2
        probs = self.soft_prob(distances, smooth)
        encodings = probs
        # print("---1--", probs.shape, probs)

        # 4. Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)  # .view(input_shape)

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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_lin = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        if debug:
            print("VQ_Embedding", self._embedding)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):

        loss_vq, loss, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                  torch.tensor(0), torch.tensor(0)
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


        loss_vq, loss, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                  torch.tensor(0), torch.tensor(0)
        quantized = torch.reshape(inputs, first_shape).contiguous()

        xxx = torch.sum(org -quantized)

        return loss, quantized, perplexity_vq, encodings

        # Calculate distances
        if debug:
            print("(torch.sum(flat_input ** 2, dim=1, keepdim=True)", torch.sum(flat_input ** 2, dim=1, keepdim=True).shape)
            print("torch.sum(self._embedding.weight ** 2, dim=1)", torch.sum(self._embedding.weight ** 2, dim=1).shape)
            print("2 * torch.matmul(flat_input, self._embedding.weight.t()))", (2 * torch.matmul(flat_input, self._embedding.weight.t())).shape)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # print("Idices: ", encoding_indices)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_lin = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = torch.hstack((inputs[0], inputs[1]))
        input_shape = inputs.shape

        inputs = self.pre_lin(inputs)

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

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
        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor
    """

    def __init__(self, n_channels, n_classes, vec_len, num_group, num_sample, normalize=False):
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
            raise ValueError('num of embeddings in each group should be an integer')
        self._num_classes_per_group = int(self.n_classes / self._num_group)

        # self.embedding0 = nn.Parameter(
        #     torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self._num_embeddings = n_classes
        self._embedding_dim = 400
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.after_update()

    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self._embedding / self._embedding.norm(dim=2, keepdim=True)
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
            d = (torch.sum(x1 ** 2, dim=1, keepdim=True)
             + torch.sum(embedding.weight ** 2, dim=1)
             - 2 * torch.matmul(x1, embedding.weight.t()))

            # Compute the group-wise distance
            d_group = torch.zeros(x1_chunk.shape[0], self._num_group).to(torch.device('cuda'))
            for i in range(self._num_group):
                d_group[:, i] = torch.mean(
                    d[:, i * self._num_classes_per_group: (i + 1) * self._num_classes_per_group], 1)
            degrup_numpy = d_group.detach().cpu().numpy()
            # Find the nearest group
            index_chunk_group = d_group.argmin(dim=1).unsqueeze(1)

            # Generate mask for the nearest group
            index_chunk_group = index_chunk_group.repeat(1, self._num_classes_per_group)
            index_chunk_group = torch.mul(self._num_classes_per_group, index_chunk_group)
            idx_mtx = torch.LongTensor([x for x in range(self._num_classes_per_group)]).unsqueeze(0).cuda()
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
            prob_chunks.append(p[:, :self._num_sample])
            index_chunks.append(idx[:, :self._num_sample])

        index = torch.cat(index_chunks, dim=0)
        prob_dist = torch.cat(prob_chunks, dim=0)
        prob_dist = F.normalize(prob_dist, p=1, dim=1)
        # index: (N*samples, n_channels) long tensor
        if True:  # compute the entropy
            hist = index[:, 0].float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
            # logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        entropy = torch.tensor(entropy)
        index1 = (index + self.offset)
        # index1: (N*samples*n_channels) long tensor
        output_list = []
        for i in range(self._num_sample):
            output_list.append(torch.mul(embedding.weight.view(-1, embedding.weight.size(1)).index_select(dim=0, index=index1[:, i]),
                                         prob_dist[:, i].unsqueeze(1).detach()))

        output_cat = torch.stack(output_list, dim=2)
        output_flat = torch.sum(output_cat, dim=2)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = F.mse_loss( x.detach() , output) #.float().norm(dim=2).pow(2)
        out2 = F.mse_loss(x , output.detach()) #.float()) #.norm(dim=2).pow(2) + (x - x0).float().norm(dim=2).pow(2)
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
    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self._embedding.size(2))
                self._embedding.mul_(target_norm / self._embedding.norm(dim=2, keepdim=True))