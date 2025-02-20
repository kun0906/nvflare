"""
    https://medium.com/@j.calzaretta.ai/exploring-diffusion-models-a-hands-on-approach-with-mnist-baf79aa4d195

    https://github.com/jcalz23/diffusion_diy/blob/main/src/attention.py

    https://github.com/jcalz23/diffusion_diy/blob/main/src/visualize_results.ipynb

"""
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
from torchmetrics.functional.image import lpips


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)

    def forward(self, tokens, context=None):
        if self.self_attn:
            Q, K, V = self.query(tokens), self.key(tokens), self.value(tokens)
        else:
            Q, K, V = self.query(tokens), self.key(context), self.value(context)

        scoremats = torch.einsum('bth,bsh->bts', Q, K)
        attnmats = F.softmax(scoremats, dim=1)
        ctx_vecs = torch.einsum("bts,bsh->bth", attnmats, V)
        return ctx_vecs


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super(TransformerBlock, self).__init__()
        self.attn_self = CrossAttention(hidden_dim, hidden_dim)
        self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x, context=None):
        x = self.attn_self(self.norm1(x)) + x
        x = self.attn_cross(self.norm2(x), context=context) + x
        x = self.ffn(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super(SpatialTransformer, self).__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x + x_in


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm, trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# @title A handy training function
def train_diffusion_model(dataset,
                          score_model,
                          marginal_prob_std_fn,
                          n_epochs=100,
                          batch_size=1024,
                          lr=10e-4,
                          model_name="transformer"):
    # Print model architecture size
    total_params = sum(p.numel() for p in score_model.parameters())
    trainable_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)

    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("-----------------------------")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(score_model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm(tqdm_epoch):
        avg_loss = 0.
        num_items = 0
        for x, y in tqdm(data_loader):
            x = x.to(device)
            # if "ldm" in model_name:
            #     loss = loss_fn_cond_ldm(score_model, x, y, marginal_prob_std_fn)
            # else:
            loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss:5f}")
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), f'ckpt_{model_name}.pth')


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


def marginal_prob_std(t, sigma):
    t = t.to(device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma ** t.to(device)


import torch


def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-5):
    """
    Computes the loss for a conditional denoising diffusion probabilistic model (DDPM).

    Args:
        model: The neural network model that predicts the score (i.e., the gradient of the log probability).
        x (torch.Tensor): The original data samples (e.g., images) with shape (batch_size, channels, height, width).
        y (torch.Tensor): The conditional information (e.g., class labels or other auxiliary data).
        marginal_prob_std (function): A function that returns the standard deviation of the noise at a given time step.
        eps (float, optional): A small value to ensure numerical stability. Default is 1e-5.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """

    # Sample a random time step for each sample in the batch.
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

    # Sample random noise from a standard normal distribution with the same shape as the input.
    z = torch.randn_like(x)

    # Compute the standard deviation of the noise at the sampled time step.
    std = marginal_prob_std(random_t)

    # Perturb the input data with the sampled noise, scaled by the computed standard deviation.
    perturbed_x = x + z * std[:, None, None, None]

    # Predict the score (denoising direction) using the model.
    # The model takes the perturbed data, the time step, and the conditional information as inputs.
    score = model(perturbed_x, random_t, y=y)

    # Compute the loss as the mean squared error between the predicted score and the true noise,
    # weighted by the standard deviation.
    # loss = torch.mean(torch.sum((score * std[:, None, None, None] - z)**2, dim=(1,2,3)))
    loss = F.mse_loss(score * std[:, None, None, None], -z, reduction='mean')

    return loss


from torchvision.models import vgg16


def loss_fn_cond_lpips(model, x, y, marginal_prob_std, eps=1e-5, lpips_weight=0.1):
    """
    Computes the loss for a conditional denoising diffusion probabilistic model (DDPM)
    with additional LPIPS perceptual loss.

    Args:
        model: The neural network model that predicts the score.
        x (torch.Tensor): The original data samples.
        y (torch.Tensor): The conditional information.
        marginal_prob_std (function): A function that returns the standard deviation of the noise.
        eps (float, optional): A small value to ensure numerical stability. Default is 1e-5.
        lpips_weight (float, optional): Weight for the LPIPS loss. Default is 0.1.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """

    # Initialize LPIPS loss function
    lpips_fn = lpips.LPIPS(net='vgg').to(x.device)

    # Sample random time step and noise
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)

    # Compute standard deviation and perturb input
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]

    # Predict the score
    score = model(perturbed_x, random_t, y=y)

    # Compute MSE loss
    mse_loss = F.mse_loss(score * std[:, None, None, None], -z, reduction='mean')

    # Compute LPIPS loss
    # We need to ensure the input is in the right format for LPIPS (3 channels)
    x_3ch = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
    perturbed_x_3ch = perturbed_x.repeat(1, 3, 1, 1) if perturbed_x.shape[1] == 1 else perturbed_x
    lpips_loss = lpips_fn(x_3ch, perturbed_x_3ch).mean()

    # Combine losses
    total_loss = mse_loss + lpips_weight * lpips_loss

    return total_loss


def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           num_steps,
                           batch_size=64,
                           x_shape=(1, 28, 28),
                           device='cuda',
                           eps=1e-3, y=None):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) \
             * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    # Do not include any noise in the last sampling step.
    return mean_x

class UNet_Tranformer(nn.Module):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256,
               text_dim=256, nClass=10):
        super().__init__()
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Encoding layers
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim)

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn4 = SpatialTransformer(channels[3], text_dim)

        # Decoding layers
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1)

        self.act = nn.SiLU()
        self.marginal_prob_std = marginal_prob_std
        self.cond_embed = nn.Embedding(nClass, text_dim)

    def forward(self, x, t, y=None):
        # Embed time and text
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)
        
        # Encoding
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h3 = self.attn3(h3, y_embed)
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))
        h4 = self.attn4(h4, y_embed)

        # Decoding
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(h + h3) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(h + h2) + self.dense7(embed)))
        h = self.tconv1(h + h1)

        # Normalize predicted noise by std at time t
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
    
class AutoEncoder(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, channels=[4, 8, 32],):
        """Initialize a time-dependent score-based network.
        Args:
            channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        # Encoding layers where the resolution decreases
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[2]),
            ) #nn.SiLU(),
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=True, output_padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[0], 1, 3, stride=1, bias=True),
            nn.Sigmoid(),
            )

    def forward(self, x):
        output = self.decoder(self.encoder(x))
        return output
    

class Latent_UNet_Tranformer(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[4, 64, 128, 256], embed_dim=256,
                 text_dim=256, nClass=10):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[1])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[2])
        self.gnorm2 = nn.GroupNorm(4, num_channels=channels[2])
        self.attn2 = SpatialTransformer(channels[2], text_dim)
        self.conv3 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[3])
        self.gnorm3 = nn.GroupNorm(4, num_channels=channels[3])
        self.attn3 = SpatialTransformer(channels[3], text_dim)

        self.tconv3 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, )
        self.dense6 = Dense(embed_dim, channels[2])
        self.tgnorm3 = nn.GroupNorm(4, num_channels=channels[2])
        self.attn6 = SpatialTransformer(channels[2], text_dim)
        self.tconv2 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)     # + channels[2]
        self.dense7 = Dense(embed_dim, channels[1])
        self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[1])
        self.tconv1 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=1) # + channels[1]

        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        self.cond_embed = nn.Embedding(nClass, text_dim)

    def forward(self, x, t, y=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)
        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h2 = self.attn2(h2, y_embed)
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h3 = self.attn3(h3, y_embed)

        # Decoding path
        ## Skip connection from the encoding path
        h = self.tconv3(h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.attn6(h, y_embed)
        h = self.tconv2(h + h2)
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h