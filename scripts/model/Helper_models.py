import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F



class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.feat_extractor = nn.Sequential(
        self.al1 = nn.Conv1d(1, 16, 15, stride=5, padding=1600)
        self.al2 = nn.BatchNorm1d(16)
        self.al3 = nn.LeakyReLU(0.3, inplace=True)
        self.al4 = nn.Conv1d(16, 32, 15, stride=6)
        self.al5 = nn.BatchNorm1d(32)
        self.al6 = nn.LeakyReLU(0.3, inplace=True)
        self.al7 = nn.Conv1d(32, 64, 15, stride=6)
        self.al8 = nn.BatchNorm1d(64)
        self.al9 = nn.LeakyReLU(0.3, inplace=True)
        self.al10 = nn.Conv1d(64, 128, 20, stride=6)
        self.al11 = nn.Conv1d(128, 200, 15, stride=8)
        # )

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        # out = self.feat_extractor(wav_data)
        out1 = self.al1(wav_data)
        out2 = self.al2(out1)
        out3 = self.al3(out2)
        out4 = self.al4(out3)
        out5 = self.al5(out4)
        out6 = self.al6(out5)
        out7 = self.al7(out6)
        out8 = self.al8(out7)
        out9 = self.al9(out8)
        out10 = self.al10(out9)
        out11 = self.al11(out10)

        return out11.transpose(1, 2)  # to (batch x seq x dim)

def spec_conv1d(n_layer=3, n_channel=[64, 32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2]):
    """
    Construction of conv. layers. Note the current implementation always effectively turn to 1-D conv,
    inspired by https://arxiv.org/pdf/1704.04222.pdf.
    :param n_layer: number of conv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 ).
            The first channel is the number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :return: an object (nn.Sequential) constructed of specified conv. layers
    TODO:
        [x] directly use nn.Conv1d for implementation
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg

    # construct layers
    conv_layers = []
    for i in range(n_layer):
        in_channel, out_channel = n_channel[i:i + 2]
        conv_layers += [
            nn.Conv1d(in_channel, out_channel, filter_size[i], stride[i]),
            nn.ReLU(),
            nn.BatchNorm1d(out_channel)
        ]

    return nn.Sequential(*conv_layers)

def fc(n_layer, n_channel, activation='tanh', batchNorm=True):
    """
    Construction of fc. layers.
    :param n_layer: number of fc. layers
    :param n_channel: in/output number of neurons for each layer ( len(n_channel) = n_layer + 1 )
    :param activation: allow either 'tanh' or None for now
    :param batchNorm: True|False, indicate apply batch normalization or not
    TODO:
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    assert activation in [None, 'tanh'], "Only implement 'tanh' for now"

    fc_layers = []
    for i in range(n_layer):
        layer = [nn.Linear(n_channel[i], n_channel[i + 1])]
        if batchNorm:
            layer.append(nn.BatchNorm1d(n_channel[i + 1]))
        if activation:
            layer.append(nn.Tanh())
        fc_layers += layer

    return nn.Sequential(*fc_layers)

class WavEncoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = [128, 32]
        self.n_freqBand, self.n_contextWin = self.input_size
        n_convChannel = [32, 16, 8]
        n_convLayer = 3
        filter_size = [1, 3, 3]
        stride = [1, 2, 2]
        n_fcLayer = 1
        n_fcChannel = [200]
        self.encoder = spec_conv1d(n_convLayer, [self.n_freqBand] + n_convChannel, filter_size, stride)
        self.flat_size, self.encoder_outputSize = self._infer_flat_size()
        self.encoder_fc = fc(n_fcLayer, [self.flat_size, *n_fcChannel], activation='tanh', batchNorm=True)

    def _infer_flat_size(self):
        encoder_output = self.encoder(torch.ones(1, *self.input_size))
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]

    def forward(self, x):
        h = self.encoder(x)
        h2 = self.encoder_fc(h.view(-1, self.flat_size))

        return h2 #.transpose(1, 2)  # to (batch x seq x dim)


Audio_Features = True
both = False
class EncoderRNN_With_Audio(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5, pre_trained_embedding=None):
        super(EncoderRNN_With_Audio, self).__init__()
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


        if Audio_Features:
            self.combine_lin = nn.Linear(embed_size * 2, embed_size)
            self.audio_encoder = WavEncoder2()
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=self.dropout, bidirectional=True)
        if both:
            self.audio_encoder = WavEncoder_tri()
            self.gru = nn.GRU(embed_size+hidden_size, hidden_size, n_layers,
                              dropout=self.dropout, bidirectional=True)


        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, input_text, input_lengths, input_audio, hidden=None):
        '''
        :param input_text:
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
        '''

        embedded = self.embedding(input_text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        '''

        if Audio_Features or both:
            # Todo: we may need do somepreprocessing.
            audio_shape = input_audio.shape # [B, Seq_lenght, 1D-signal]

            audio_raw_for_now = False
            if  audio_raw_for_now:
                input_audio = torch.reshape(input_audio, (-1, audio_shape[2])) # [B*Seq_len, 1D_signal]

            else:
                input_audio = torch.reshape(input_audio, (-1, audio_shape[-2],  # [B*Seq_len, freq_band, chunk_len]
                                                          audio_shape[-1]))
            audio_feat = self.audio_encoder(input_audio)
            audio_feat = torch.reshape(audio_feat, (audio_shape[1], audio_shape[0], -1 ))
            if both:
                embedded = self.embedding(input_text)
                packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
                packed = torch.cat((packed, audio_feat), dim=2)
            # packed = self.combine_lin(packed)
            else:
                packed = audio_feat

        outputs, hidden = self.gru(packed, hidden)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden





# ----------------------------------------------------------
# Trimodal paper
# ----------------------------------------------------------

class WavEncoder_tri(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
        )
        self.out_layer = nn.Linear(32*15, 200)
    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)

        out = torch.reshape(out, (out.shape[0], -1))
        out = self.out_layer(out)
        return out #.transpose(1, 2)  # to (batch x seq x dim)

# ----------------------------------------------------------
from model.tcn import TemporalConvNet
class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=False)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layers
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 32)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0







# Attention is all you need
'''
class Encoder_ATX(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer_ATX(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer_ATX(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer_ATX(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Decoder_ATX(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention


class DecoderLayer_ATX(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention


class Seq2Seq_ATX(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention


class DNN(nn.Module):
    def __init__(self, n_layers, hidden_units, input_dim, output_dim, device):
        super(DNN, self).__init__()
        fc = []
        self.n_layers = n_layers
        for i in range(n_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_units).to(device))
            elif i == n_layers-1:
                fc.append(nn.Linear(hidden_units, output_dim).to(device))
            else:
                fc.append(nn.Linear(hidden_units, hidden_units).to(device))
        self.fc = fc

    def forward(self, x):
        for i in range(self.n_layers):
            if i == self.n_layers-1:
                 x = F.sigmoid(self.fc[i](x))
            else:
                 x = F.relu(self.fc[i](x))
        return x
'''