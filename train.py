# %%
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusion_utilities import *
from model import ContextUnet
from dataset import CustomDataset

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64 # 64 hidden dimension feature
n_cfeat = 5 # context vector is of size 5
height = 16 # 16x16 image
save_dir = './weights/'

# training hyperparameters
batch_size = 100
n_epoch = 32
lrate=1e-3

# %%
# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1

# %%
# construct model
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

# load dataset and construct optimizer
dataset = CustomDataset("./data/sprites_1788_16x16.npy", "./data/sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

def perturb_input(x, t, noise):
    """
    A function that perturbs the input `x` based on the value of `t` and a noise factor.
    
    Parameters:
    x : input array
    t : index value
    noise : noise factor
    
    Returns:
    The perturbed input array.
    """
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
# %%

def apply_context_masking(context, device):
    """
    Applies random masking to the context vectors to enhance model robustness.
    
    This method randomly sets elements of the context vector to zero with a probability of 0.1,
    serving as a regularization technique. It encourages the model to not solely rely on context 
    information, promoting better generalization by learning more from the image content itself.
    When context is present, it aids in refining model predictions.

    Parameters:
    - context (Tensor): The context vectors for the batch of images.
    - device (torch.device): The device to perform the operation on.

    Returns:
    - Tensor: The context vectors with applied masking.
    """
    context_mask = torch.bernoulli(torch.zeros(context.shape[0]) + 0.9).to(device)
    return context * context_mask.unsqueeze(-1)

nn_model.train()

for ep in range(n_epoch):
    print(f'epoch {ep}')
    
    # linearly decay learning rate
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
    
    pbar = tqdm(dataloader, mininterval=2 )
    for x, c in pbar:   # x: images  c: context
        optim.zero_grad()
        x = x.to(device)
        c = c.to(x)
        apply_context_masking(c, device)
        
        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
        x_pert = perturb_input(x, t, noise)
        
        # use network to recover noise
        pred_noise = nn_model(x_pert, t / timesteps, c=c)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        
        optim.step()

    # save model periodically
    if ep%4==0 or ep == int(n_epoch-1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(nn_model.state_dict(), save_dir + f"context_model_{ep}.pth")
        print('saved model at ' + save_dir + f"context_model_{ep}.pth")