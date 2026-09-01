
#
# Clear -> Thick -> Fog -> Thin -> Local -> repeat
#
# Clear:
#   - original image
#   - NO cloud
#   - NO normalization in final saved image
#   - cloud mask  = zero
#   - shadow mask = zero
#
# Thick:
#   - cloud + shadow
#
# Local:
#   - cloud + shadow
#
# Thin:
#   - cloud only
#   - shadow mask saved as zero
#
# Fog:
#   - cloud only
#   - shadow mask saved as zero
#


import os
import glob
import sys
import inspect
import numpy as np
import torch
import rasterio


# ============================================================
# IMPORT SATELLITE CLOUD GENERATOR
# ============================================================

sys.path.append("./SatelliteCloudGenerator")

from src import add_cloud, add_cloud_and_shadow


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"256x256_Dfc2020/s2_0/"

OUTPUT_DIR = r"256x256_Dfc2020/CLOUDY_TRAINING_S2"

MASK_DIR_CLOUD = os.path.join(
    OUTPUT_DIR,
    "cloud_masks"
)

MASK_DIR_SHADOW = os.path.join(
    OUTPUT_DIR,
    "shadow_masks"
)


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def norm01_percentile(chw, p=99):
    """
    Temporary normalization used ONLY for cloud synthesis.

    Input:
        chw : (C,H,W)

    Returns:
        normalized image
        scale

    NOTE:
    One common scale is computed over the complete image.
    The final cloudy image is multiplied by this scale again.
    """

    scale = np.percentile(chw, p)

    normalized = np.clip(
        chw / (scale + 1e-6),
        0.0,
        1.0
    )

    return normalized.astype(np.float32), float(scale)


def channel_magnitude_torch(B, C, device):
    """
    Channel magnitude vector used by CSM.

    Different bands receive slightly different magnitudes.
    """

    values = torch.linspace(
        1.15,
        0.85,
        C,
        device=device,
        dtype=torch.float32
    )

    return values.unsqueeze(0).repeat(B, 1)


def sample_range(value):
    """
    If [min,max] is provided, randomly sample one value.
    Otherwise use the fixed value.
    """

    if isinstance(value, (list, tuple)) and len(value) == 2:

        return float(
            np.random.uniform(
                value[0],
                value[1]
            )
        )

    return float(value)


def sample_int_range(value):
    """
    Random integer sampling for locality_degree.
    """

    if isinstance(value, (list, tuple)) and len(value) == 2:

        return int(
            np.random.randint(
                int(value[0]),
                int(value[1]) + 1
            )
        )

    return int(value)


def mask_to_2d(mask):
    """
    Convert cloud/shadow mask into (H,W).

    Possible inputs:
        (B,C,H,W)
        (C,H,W)
        (1,H,W)
        (H,W)
    """

    if mask is None:
        return None

    m = mask.detach().cpu()

    # (B,C,H,W)
    if m.ndim == 4:
        m = m[0]

    # (C,H,W)
    if m.ndim == 3:
        m = m.max(dim=0).values

    return m.numpy().astype(np.float32)


def call_with_supported_args(fn, input_tensor, **kwargs):
    """
    Only send arguments supported by the installed
    SatelliteCloudGenerator version.
    """

    sig = inspect.signature(fn)

    allowed = set(sig.parameters.keys())

    for first_arg in [
        "input",
        "x",
        "img",
        "image"
    ]:
        allowed.discard(first_arg)

    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in allowed
    }

    return fn(
        input_tensor,
        **filtered_kwargs
    )


def write_geotiff(path, arr_chw, ref_profile):
    """
    Save multiband float32 GeoTIFF.
    """

    profile = ref_profile.copy()

    profile.update(
        dtype=rasterio.float32,
        count=arr_chw.shape[0],
        compress="deflate"
    )

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            arr_chw.astype(np.float32)
        )


def write_mask(path, mask_hw, ref_profile):
    """
    Save single-band mask GeoTIFF.
    """

    profile = ref_profile.copy()

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="deflate"
    )

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            mask_hw.astype(np.float32),
            1
        )


def zero_mask(height, width):
    """
    Create empty mask.
    """

    return np.zeros(
        (height, width),
        dtype=np.float32
    )


# ============================================================
# CLOUD CONFIGURATIONS

configs = {

    # --------------------------------------------------------
    # THICK
    # --------------------------------------------------------
    "Thick": {

        "min_lvl": 0.0,

        "max_lvl": 1.0,

        "threshold": [0.0, 0.2],

        "locality_degree": 1,

        "decay_factor": 1.0,

        "cloud_color": True,

        "channel_offset": 2,

        "blur_scaling": 2.0,

        "with_shadow": True,
    },


    # --------------------------------------------------------
    # LOCAL
    # --------------------------------------------------------
    "Local": {

        "min_lvl": 0.0,

        "max_lvl": 1.0,

        "threshold": [0.0, 0.2],

        "locality_degree": [2, 4],

        "decay_factor": 1.0,

        "cloud_color": True,

        "channel_offset": 2,

        "blur_scaling": 2.0,

        "with_shadow": True,
    },


    # --------------------------------------------------------
    # THIN
    # --------------------------------------------------------
    "Thin": {

        "min_lvl": [0.0, 0.1],

        "max_lvl": [0.4, 0.7],

        "threshold": 0.0,

        "locality_degree": [1, 3],

        "decay_factor": 1.0,

        "cloud_color": True,

        "channel_offset": 2,

        "blur_scaling": 2.0,

        "with_shadow": False,
    },


    # --------------------------------------------------------
    # FOG
    # --------------------------------------------------------
    "Fog": {

        "min_lvl": [0.3, 0.6],

        "max_lvl": [0.6, 0.7],

        "threshold": 0.0,

        "locality_degree": 1,

        "decay_factor": 1.0,

        "cloud_color": True,

        "channel_offset": 2,

        "blur_scaling": 2.0,

        "with_shadow": False,
    },
}


# ============================================================
# EXACT CYCLING ORDER
# ============================================================

ORDER = [
    "Clear",
    "Thick",
    "Fog",
    "Thin",
    "Local"
]


# ============================================================
# SHADOW PARAMETER
# ============================================================

SHADOW_MAX_LVL = [0.2, 0.5]


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one(
    tif_path,
    out_path,
    cloud_type
):

    # --------------------------------------------------------
    # READ ORIGINAL IMAGE
    # --------------------------------------------------------

    with rasterio.open(tif_path) as src:

        arr = src.read().astype(
            np.float32
        )

        profile = src.profile

    C, H, W = arr.shape


    # ========================================================
    # CLEAR SAMPLE
    # ========================================================

    if cloud_type == "Clear":

        # Keep original image exactly in original scale
        write_geotiff(
            out_path,
            arr,
            profile
        )

        # No cloud and no shadow
        z = zero_mask(H, W)

        write_mask(
            os.path.join(
                MASK_DIR_CLOUD,
                os.path.basename(out_path)
            ),
            z,
            profile
        )

        write_mask(
            os.path.join(
                MASK_DIR_SHADOW,
                os.path.basename(out_path)
            ),
            z,
            profile
        )

        return True


    # ========================================================
    # CLOUDY SAMPLE
    # ========================================================

    cfg = configs[cloud_type]


    # --------------------------------------------------------
    # Temporary normalization to [0,1]
    # --------------------------------------------------------

    arr_normalized, scale = norm01_percentile(
        arr,
        p=99
    )

    x = torch.from_numpy(
        arr_normalized
    ).unsqueeze(0)

    B, C, H, W = x.shape


    # --------------------------------------------------------
    # CHANNEL MAGNITUDE / CSM
    # --------------------------------------------------------

    channel_magnitude = channel_magnitude_torch(
        B,
        C,
        x.device
    )


    # --------------------------------------------------------
    # Sample cloud parameters ONCE PER IMAGE
    # --------------------------------------------------------

    min_lvl = sample_range(
        cfg["min_lvl"]
    )

    max_lvl = sample_range(
        cfg["max_lvl"]
    )

    threshold = sample_range(
        cfg["threshold"]
    )

    locality_degree = sample_int_range(
        cfg["locality_degree"]
    )

    decay_factor = float(
        cfg["decay_factor"]
    )


    # --------------------------------------------------------
    # Cloud generator arguments
    # --------------------------------------------------------

    base_kwargs = dict(

        return_cloud=True,

        noise_type="perlin",

        min_lvl=(
            min_lvl,
            min_lvl
        ),

        max_lvl=(
            max_lvl,
            max_lvl
        ),

        decay_factor=decay_factor,

        locality_degree=locality_degree,

        clear_threshold=threshold,

        channel_magnitude=channel_magnitude,

        cloud_color=cfg["cloud_color"],

        channel_offset=cfg["channel_offset"],

        blur_scaling=cfg["blur_scaling"],
    )


    # ========================================================
    # THICK / LOCAL
    # CLOUD + SHADOW
    # ========================================================

    if cfg["with_shadow"]:

        cloudy, cloud_mask, shadow_mask = (
            call_with_supported_args(

                add_cloud_and_shadow,

                x,

                shadow_max_lvl=SHADOW_MAX_LVL,

                **base_kwargs
            )
        )


    # ========================================================
    # THIN / FOG
    # CLOUD ONLY
    # ========================================================

    else:

        cloudy, cloud_mask = (
            call_with_supported_args(

                add_cloud,

                x,

                **base_kwargs
            )
        )

        shadow_mask = None


    # ========================================================
    # RETURN CLOUDY IMAGE TO ORIGINAL SCALE
    # ========================================================

    cloudy_raw = (

        cloudy[0]
        .detach()
        .cpu()
        .numpy()

        * scale

    ).astype(np.float32)


    # --------------------------------------------------------
    # SAVE CLOUDY IMAGE
    # --------------------------------------------------------

    write_geotiff(
        out_path,
        cloudy_raw,
        profile
    )


    # ========================================================
    # CLOUD MASK
    # ========================================================

    cloud_mask_2d = mask_to_2d(
        cloud_mask
    )

    if cloud_mask_2d is None:

        cloud_mask_2d = zero_mask(
            H,
            W
        )


    # ========================================================
    # SHADOW MASK
    # ========================================================

    shadow_mask_2d = mask_to_2d(
        shadow_mask
    )

    if shadow_mask_2d is None:

        shadow_mask_2d = zero_mask(
            H,
            W
        )


    # --------------------------------------------------------
    # SAVE MASKS
    # --------------------------------------------------------

    write_mask(

        os.path.join(
            MASK_DIR_CLOUD,
            os.path.basename(out_path)
        ),

        cloud_mask_2d,

        profile
    )


    write_mask(

        os.path.join(
            MASK_DIR_SHADOW,
            os.path.basename(out_path)
        ),

        shadow_mask_2d,

        profile
    )


    return True


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dir(
        OUTPUT_DIR
    )

    ensure_dir(
        MASK_DIR_CLOUD
    )

    ensure_dir(
        MASK_DIR_SHADOW
    )


    tif_list = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "*.tif"
            )
        )
    )


    if len(tif_list) == 0:

        raise RuntimeError(
            f"No .tif files found in {INPUT_DIR}"
        )


    print(
        f"Number of input images: {len(tif_list)}"
    )

    print(
        "Generation order:"
    )

    print(
        "Clear -> Thick -> Fog -> Thin -> Local -> repeat"
    )

    print()


    # ========================================================
    # PROCESS ALL IMAGES
    # ========================================================

    for i, tif_path in enumerate(tif_list):

        cloud_type = ORDER[
            i % len(ORDER)
        ]

        filename = os.path.basename(
            tif_path
        )

        out_path = os.path.join(
            OUTPUT_DIR,
            filename
        )


        # Print first five samples
        if i < 5:

            print(
                f"[SAMPLE {i+1}] "
                f"{filename} -> {cloud_type}"
            )


        process_one(
            tif_path,
            out_path,
            cloud_type
        )


    print()
    print("========================================")
    print("DONE")
    print("========================================")

    print(
        "Images:",
        OUTPUT_DIR
    )

    print(
        "Cloud masks:",
        MASK_DIR_CLOUD
    )

    print(
        "Shadow masks:",
        MASK_DIR_SHADOW
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
