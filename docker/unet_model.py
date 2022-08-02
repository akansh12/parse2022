from monai.networks.nets import UNet
from monai.networks.layers import Norm
from collections import OrderedDict
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

def Unet_model(path2weights):
    
    UNet_meatdata = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH
    )
    unet = UNet(**UNet_meatdata)
    
    if device == torch.device("cpu"):
        state_dict = torch.load(path2weights, map_location='cpu')
    else:
        state_dict = torch.load(path2weights)
        
    for keyA, keyB in zip(state_dict, unet.state_dict()):
        state_dict = OrderedDict((keyB if k == keyA else k, v) for k, v in state_dict.items())
    unet.load_state_dict(state_dict)
    unet = unet.to(device)       
    
    return unet






