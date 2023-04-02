"""
"""


import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

class DAE_Network(nn.Module):
    """Autoencoder neural net with noise.

    Neural net for frame-level generation of gestures.
    Uses Linear layers for both encoder and decoder.
    Adds noise during the training process using a Dropout layer.

    Attributes:
        dropout: A PyTorch Dropout layer with a 20% chance of dropout.
        encoder: A PyTorch Linear Layer with a ReLU activation function.
        decoder: A PyTorch Linear layer.
    """

    def __init__(self, motion_dim: int, latent_dim: int):
        """Initialize with the dataset and the latent code space dimensions.

        For 'latent_dim', there are sentinel values of -1 and -2.
        Value of -1 indicates to skip the neural net for ablation-study.
        Value of -2 indicates Linear transformation to size 200 latent space with 30% dropout layer.

        Args:
            motion_dim: The original integer dimension of the training data.
            latent_dim: The integer size of the latent code space.
        """
        super(DAE_Network, self).__init__()
        print('init DAE')

        self.dropout = nn.Dropout(0.2)

        # To skip this network for ablation-study
        if latent_dim == -1:
            self.encoder = None
            self.decoder = None
            return

        # Linear transformation
        if latent_dim == -2:
            self.encoder = nn.Sequential(nn.Linear(motion_dim, 200),)
            self.decoder = nn.Sequential(nn.Linear(200, motion_dim),)
            self.dropout = nn.Dropout(0.3)
            return

        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, motion_dim),
        )


    def forward(self, x: torch.Tensor, get_latent: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the net.

        Only a single epoch is needed (Encoder and Decoder are Linear layers only).
        If sentinel value of -1 was used during initialization then
        early termination of method occurs.

        Args:
            x: The training data as a PyTorch Tensor.
            get_latent: Boolean that indicates whether to return the latent space between encoder and decoder.

        Returns:
            Either a single PyTorch Tensor or a Tuple containing 2 PyTorch Tensors.

            Case 1 - Encoder is None (ie. -1 was used during initialization):
                Return the input data only.
            Case 2 - Encoder is None and get_latent is True:
                Return the input data as both Tensors.
            Case 3 - Encoder is not None:
                Return the output data as normal.
            Case 4 - Encoder is not None and get_latent is True:
                Return the output data and the latent space as 2 Tensors respectively.
        """
        if self.encoder is None:
            if get_latent:
                return x, x
            else:
                return x
        input_data = torch.squeeze(x)
        partial_input_data: torch.Tensor = self.dropout(input_data)
        latent_space: torch.Tensor = self.encoder(partial_input_data)
        latent_copy = latent_space.detach().clone()
        output_data: torch.Tensor = self.decoder(latent_space)
        format_output_data = torch.unsqueeze(output_data, 2)
        if get_latent:
            return format_output_data, latent_copy
        else:
            return format_output_data

# Vector quantization at the frame level
class VQ_Frame(nn.Module):
    """

    Attributes:
        encoder:
        decoder:
        bachnorm:
        skip_vq:
        vq_components:
        commitment_cost
        decay:
        vq_layer:
        gs_soft:
        log_scale:
        dropout:
        vae:
    """
    def __init__(self, motion_dim: int, latent_dim: int, vae: bool, vq_components: int):
        super(VQ_Frame, self).__init__()
        print("init VQ_DAE")


        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, latent_dim),
            # nn.Tanh(),
            # nn.Linear(motion_dim-10, motion_dim-30),
            # nn.Tanh(),
        )
        nn.init.xavier_normal_(self.encoder[0].weight)

        # Todo: test
        self.bachnorm = nn.BatchNorm1d(latent_dim)

        self.skip_vq = False
        self.vq_components = vq_components
        self.commitment_cost = 0.25
        decay = 0.99
        if decay > 0.0:
            self.vq_layer = VQ_Payam_EMA(self.vq_components,
                                     latent_dim, self.commitment_cost, decay)
        else:
            self.vq_layer = VQ_Payam(self.vq_components,
                                     latent_dim, self.commitment_cost)


        # GS-Soft Vector Quantization
        self.gs_soft = False
        if self.gs_soft:
            self.vq_layer = VQ_Payam_GSOFT(self.vq_components,
                                         latent_dim, self.commitment_cost, decay)
            self.log_scale = nn.Parameter(torch.Tensor([0.0]))


        self.decoder = nn.Sequential(
            # nn.Linear(motion_dim-30, motion_dim-10),
            # nn.Tanh(),
            nn.Linear(latent_dim, motion_dim),
        )
        nn.init.xavier_normal_(self.decoder[0].weight)
        self.dropout = nn.Dropout()

        self.vae = vae
        if self.vae:
            self.VAE_fc_mean = nn.Linear(latent_dim, latent_dim)
            self.VAE_fc_std = nn.Linear(latent_dim, latent_dim)
            self.VAE_fc_decoder = nn.Linear(latent_dim, latent_dim)



    def reparameterize(self, mu: torch.Tensor, logVar: torch.Tensor, train: bool = True) -> torch.Tensor:
        """

        Args:
            mu:
            logVar:
            train:

        Returns:
            A PyTorch Tensor.
        """
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        # Todo: fix this for train and test time checking
        # Add noise with std and eps for training only
        if train:
            return mu + std * eps
        else:
            return mu # + std * eps

    def forward(self, x: torch.Tensor, inference: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            x:
            inference:

        Returns:
            A 3-Tuple or 5-Tuple:
        """
        # print("_________________")
        # print(self.encoder)
        # print(x.shape)
        # print("_________________")
        # 1.
        x = torch.squeeze(x)
        # print(x.shape)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.bachnorm(x)
        # 2.
        latent = x.detach().clone()
        # print("Encoded", x.shape)

        # 4.1 VAE
        if self.vae == True:
            mean = self.VAE_fc_mean(x)
            logvar = self.VAE_fc_std(x)
            z = self.reparameterize(mean, logvar)
            z = self.VAE_fc_decoder(z)
            x = z


        # 3. Vector Quantization
        if not self.skip_vq:
            loss_vq, quantized, perplexity_vq, encodings = self.vq_layer(x)
            x = quantized
        else:
            loss_vq, perplexity_vq = torch.tensor(0), torch.tensor(0)
        # 4. Decoding


        x = self.decoder(x)
        x = torch.unsqueeze(x, 2)



        # print("Decoder", x.shape)
        if inference:
            return x, latent, encodings
        else:
            if self.gs_soft:
                scale = torch.exp(self.log_scale)
                x = torch.distributions.Normal(x, scale)
            if self.vae:
                return x, loss_vq, perplexity_vq, logvar, mean
            else:
                return x, loss_vq, perplexity_vq


class VQ_Payam(nn.Module):
    """A Vector-Quantized Autoencoder neural net.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """Initialization method.

        Args:
            num_embeddings: The integer dimension of the input training data.
            embedding_dim: The integer size of the latent space.
            commitment_cost: A float percentage to apply to latent loss calculations.
        """
        super(VQ_Payam, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the net.

        Args:
            inputs: A PyTorch Tensor of training data.

        Returns:
            A 4-Tuple:
            loss: A PyTorch Tensor of loss calculations (MSELoss).
            quantized:
            perplexity:
            encodings:
        """
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)

        # 2. Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # 3. Find nearest encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings: torch.Tensor = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 4. Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)  # .view(input_shape)
        quantized = torch.reshape(quantized, inputs.shape).contiguous()

        # 5. Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss =  q_latent_loss +  self._commitment_cost * e_latent_loss

        # 6. Calculate outputs
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.contiguous(), perplexity, encodings


class VQ_Payam_EMA(nn.Module):
    """A Vector-Quantized Autoencoder neural net.

    Attributes:
        pre_linear: A PyTorch Linear layer.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float, epsilon: float = 1e-5):
        """Initialization method.

        Args:
            num_embeddings:
            embedding_dim:
            commitment_cost:
            decay:
            epsilon:
        """
        super(VQ_Payam_EMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.pre_linear = nn.Linear(self._embedding_dim, self._embedding_dim)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

        #EMA
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the net.

        Args:
            inputs:

        Returns:
            A 4-Tuple:
            loss: A PyTorch Tensor of loss calculations (MSELoss).
            quantized:
            perplexity:
            encodings:
        """
        loss_vq, loss, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                  torch.tensor(0), torch.tensor(0)
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        # flat_input = self.pre_linear(flat_input)

        # __________________________
        # 2. Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        debug = False
        if debug:
            print("VQ_flat_input", flat_input.shape)
            print("VQ_distances", distances.shape)

        # 3. Find nearest encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # for ind in encoding_indices:
        #     print("Idices: ", ind)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 4. Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)  # .view(input_shape)
        quantized = torch.reshape(quantized, inputs.shape).contiguous()

        # X. Use EMA to update the embedding vectors
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
        loss = self._commitment_cost * e_latent_loss

        # 6. Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.contiguous(), perplexity, encodings
        # ___________________________


        quantized = torch.reshape(flat_input, inputs.shape).contiguous()

        return loss, quantized, perplexity_vq, encodings

class VQ_Payam_GSOFT(nn.Module):
    """

    Attributes:
        pre_linear:
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float,
                 decay: float, epsilon: float = 1e-5):
        """Initialization function.

        Args:
            num_embeddings:
            embedding_dim:
            commitment_cost:
            decay:
            epsilon:
        """
        super(VQ_Payam_GSOFT, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.pre_linear = nn.Linear(self._embedding_dim, self._embedding_dim)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

        self._commitment_cost = commitment_cost





    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            inputs:

        Returns:
            A 4-Tuple:
            KL:
            quantized:
            perplexity:
            encodings:
        """
        loss_vq, loss, perplexity_vq, encodings = torch.tensor(0), torch.tensor(0), \
                                                  torch.tensor(0), torch.tensor(0)
        # 1. Pre-Linear
        flat_input = inputs.view(-1, self._embedding_dim)
        # flat_input = self.pre_linear(flat_input)

        # __________________________
        # 2. Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        dist = RelaxedOneHotCategorical(0.5, logits=-distances)
        encodings = dist.rsample()  # .view(N, -1, M)
        if self.training:
            encodings = dist.rsample() # .view(N=latend_dim, -1, M=#embedding)
        else:
            # 3. Find nearest encoding
            encoding_indices = torch.argmax(dist.probs, dim=1).unsqueeze(1)
            print(encoding_indices)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

            # encodings = dist.rsample()

        # 4. Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)  # .view(input_shape)
        # print(encodings.shape, encodings.sum(), encodings)
        quantized = torch.reshape(quantized, inputs.shape).contiguous()




        # 5. Loss
        KL: torch.Tensor = dist.probs * (dist.logits + math.log(self._num_embeddings))
        # print("KL_Shape", KL.shape) #128 * 100
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.sum(dim=0)
        # print("KL.sum(dim=0)", KL.shape)
        KL = KL.mean()
        # print("Kl.mean()", KL.shape, KL)


        # 6. Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return KL, quantized.contiguous(), perplexity, encodings
        # ___________________________
        quantized = torch.reshape(flat_input, inputs.shape).contiguous()
        return loss, quantized, perplexity_vq, encodings

class VAE_Network(nn.Module):
    """Variational Autoencoder neural net.

    Attributes:
        encoder:
        decoder:
        dropout:
        vae:
        VAE_fc_mean:
        VAE_fc_std
        VAE_fc_decoder:
    """
    def __init__(self, motion_dim: int, latent_dim: int):
        """Initialize with the dataset and the latent code space dimensions.

        Args:
            motion_dim: The original integer dimension of the training data.
            latent_dim: The integer size of the latent code space.
        """
        super(VAE_Network, self).__init__()
        print("init VAE_Network")
        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, latent_dim),
            nn.Tanh(),
            # nn.Linear(motion_dim-10, motion_dim-30),
            # nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(motion_dim-30, motion_dim-10),
            # nn.Tanh(),
            nn.Linear(latent_dim, motion_dim),

        )
        self.dropout = nn.Dropout()

        self.vae=True
        self.VAE_fc_mean = nn.Linear(latent_dim, latent_dim)
        self.VAE_fc_std = nn.Linear(latent_dim, latent_dim)
        self.VAE_fc_decoder = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logVar: torch.Tensor, train: bool = True) -> torch.Tensor:
        """Return 'mu' untouched or if 'train' then add noise based on 'logVar'.

        The noise added is random weight * exp('logVar'/2).

        Args:
            mu: A Tensor containing mean values.
            logVar: A Tensor of values used to derive standard deviation.
            train: A boolean whether to add the mean and random std as noise.

        Returns:
            Case 1 - 'train' is True (default):
                A Tensor with a random amount of std added as noise.
            Case 2 - 'train' is false:
                Return the input 'mu' untouched.
        """

        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        # Todo: fix this for train and test time checking
        if train:
            return mu + std * eps
        else:
            return mu  # + std * eps

    def forward(self, x: torch.Tensor, get_latent: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the net.

        Args:
            x: A Tensor containing the training data.
            get_latent: A boolean whether to return the latent space code as a Tensor.

        Returns:
            A 2-Tuple or 3-Tuple:

            Case 1 - 'get_latent' is False (default):
                Return a 2-Tuple:
                    x: A Tensor containing the output.
                    latent: A Tensor containing the latent code space.
            Case 2 - 'get_latent' is True:
                Return a 3-Tuple:
                    x: A Tensor containing the output.
                    logVar: A Tensor of base values used for std (after FC Layer).
                    mean: A Tensor of values used as mean values (after FC Layer).
        """
        # print("_________________")
        # print(self.encoder)
        # print(x.shape)
        # print("_________________")
        x = torch.squeeze(x)
        # print(x.shape)A regular autoencoder (similar to part1) with a bottleneck size of 30 dimensions, and Tanh activation function, and turning some of the inputs values to zero.
        x = self.dropout(x)
        x = self.encoder(x)
        latent = x.detach().clone()
        # print("Encoded", x.shape)

        # VAE
        mean = self.VAE_fc_mean(x)
        logvar = self.VAE_fc_std(x)
        z = self.reparameterize(mean, logvar)
        z = self.VAE_fc_decoder(z)
        x = z


        x = self.decoder(x)
        x = torch.unsqueeze(x, 2)
        # print("Decoder", x.shape)
        if get_latent:
            return x, latent
        else:
            return x, logvar, mean