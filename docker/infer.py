from swin_model import Swin_model
from unet_model import Unet_model
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
import monai
import numpy as np
from monai.inferers import sliding_window_inference
import glob
import os
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd, 
    EnsureTyped,
    EnsureType,
    Invertd,
    KeepLargestConnectedComponent,
    AddChanneld,
    ToTensord

)
test_transforms = Compose(
    [
        LoadImaged(keys=["images"]),
        EnsureChannelFirstd(keys=["images"]),
        Orientationd(keys=["images"], axcodes="LPS"),
        ScaleIntensityRanged(
            keys=["images"], a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["images"], source_key="images"),
        EnsureTyped(keys=["images"]),
    ]
)

def test_dataloader(path2input, test_transforms = test_transforms):
    root_dir = path2input
    test_files_path = sorted(glob.glob(os.path.join(root_dir, "**/*.nii.gz"), recursive = True))
    test_data = [{"images": image_name } for image_name in test_files_path]
    test_ds = Dataset(data = test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size = 1, shuffle = False)
    return test_loader



def load_model():
    swin_8687 = Swin_model("/scratch/scratch6/akansh12/challenges/parse2022/temp/selected_models/swin-again_no_back_1000hu_8655.pth")
    swin_8675 = Swin_model("/scratch/scratch6/akansh12/challenges/parse2022/temp/selected_models/swin-again_no_back_1000hu_8675.pth")
    swin_8655 = Swin_model("/scratch/scratch6/akansh12/challenges/parse2022/temp/selected_models/swin-again_no_back_1000hu_8687.pth")
    unet_8530 = Unet_model("/scratch/scratch6/akansh12/challenges/parse2022/temp/selected_models/unet_1000_hu_160_0853.pth")
    unet_8550 = Unet_model("/scratch/scratch6/akansh12/challenges/parse2022/temp/selected_models/unet_1000_hu_160_8550.pth")
    unet_8551 = Unet_model("/scratch/scratch6/akansh12/challenges/parse2022/temp/selected_models/unet_1000_hu_160_w_augmentations_8551.pth")
    
    return swin_8687, swin_8675, swin_8655, unet_8530, unet_8550, unet_8551

post_transforms_1 = Compose([AsDiscreted(keys = "pred", argmax=True)])
post_transforms_2 = Compose([
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="images",
            meta_keys=None,
            orig_meta_keys=None,
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="/scratch/scratch6/akansh12/challenges/parse2022/docker/test_outputs/", output_postfix='seg', resample=False),
    ])


loader = test_dataloader("/scratch/scratch6/akansh12/challenges/parse2022/docker/test_inputs/")
swin_8687, swin_8675, swin_8655, unet_8530, unet_8550, unet_8551 = load_model()
models = [unet_8530, unet_8550, unet_8551, swin_8687, swin_8675, swin_8655]
weights = np.array([85.30, 85.50, 85.51, 88, 87, 86.55])
weights = weights / np.sum(weights)


roi_size = (288, 288, 288)
sw_batch_size = 8
with torch.no_grad():
    for test_data in loader:
        test_data_out = 0
        for e, (model,w) in enumerate(zip(models,weights)):
            out = {}
            test_inputs = test_data["images"].to(device)
            model = model.to(device)
            if e > 2:
                roi_size = (96, 96, 96)
                sw_batch_size = 4
            out["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            out = [post_transforms_1(i) for i in decollate_batch(out)]
            test_data_out = test_data_out + w*out[0]['pred'][0]
            torch.cuda.empty_cache()
            
        test_data_out = test_data_out > 0.5
        test_data_out = test_data_out.unsqueeze(0).unsqueeze(0)
        test_data['pred'] = test_data_out
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

        


