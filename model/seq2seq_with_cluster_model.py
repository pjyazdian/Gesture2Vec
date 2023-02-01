import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math


class cluster2gesture_model(nn.Module):
    def __init__(self, args, input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):
        super(cluster2gesture_model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_frames = args.n_poses

        self.embedding = nn.Embedding(input_size, embed_size)
        assert(hidden_size==embed_size)

        self.pre_gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout,
                          batch_first=True)


        self.pre_linear = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout,
                          batch_first=True)
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_cluster, out_poses):
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
        outputs = torch.zeros_like(out_poses).to(out_poses.device)


        embedded = self.embedding(input_cluster)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        embedded = embedded.unsqueeze(1)
        encoder_output, encoder_hidden = self.pre_gru(embedded, None)

        decoder_input = torch.zeros_like(out_poses[:,0,:])
        decode_hidden = encoder_hidden
        for t in range(1, self.n_frames):
            decoder_input = self.pre_linear(decoder_input)
            decoder_input = decoder_input.unsqueeze(1) #since batch_first=true, B*82 -> b*L*82

            decoder_output, decoder_hidden = self.gru(decoder_input, decode_hidden)
            # post-fc
            decoder_output = decoder_output.squeeze(1)  # [1 x batch x hidden_size] -> [batch x hidden_size]
            output = self.out_layer(decoder_output)

            outputs[:, t, :] = output

            decoder_input = output  # next input is current prediction

        return outputs

