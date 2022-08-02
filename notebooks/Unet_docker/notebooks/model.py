from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

def model(path2weights):
    
    UNet_meatdata = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH
    )
    
    if device == torch.device("cpu"):
        unet = UNet(**UNet_meatdata).to(device)
        unet.load_state_dict(torch.load(path2weights, map_location = 'cpu'))
    else:
        unet = UNet(**UNet_meatdata).to(device)
        unet.load_state_dict(torch.load(path2weights))
        
    
    return unet


