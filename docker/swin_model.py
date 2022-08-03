from monai.networks.nets import SwinUNETR
from collections import OrderedDict
import torch

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

device = torch.device("cpu") 

def Swin_model(path2model):

    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
    )
    
    if device == torch.device("cpu"):
        state_dict = torch.load(path2model, map_location='cpu')
    else:
        state_dict = torch.load(path2model)
    for keyA, keyB in zip(state_dict, model.state_dict()):
        state_dict = OrderedDict((keyB if k == keyA else k, v) for k, v in state_dict.items())
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    return model