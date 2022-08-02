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
    Invertd
)



val_transforms = Compose(
    [
        LoadImaged(keys=["images", "labels"]),
        EnsureChannelFirstd(keys=["images", "labels"]),
        Orientationd(keys=["images", "labels"], axcodes="LPS"),
        Spacingd(keys=["images", "labels"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["images"], a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["images", "labels"], source_key="images"),
        EnsureTyped(keys=["images", "labels"]),
    ]
)