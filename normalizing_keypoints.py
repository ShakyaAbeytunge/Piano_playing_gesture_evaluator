import numpy as np
import pandas as pd
import os
import glob

INPUT_FOLDER = "keypoints"
OUTPUT_FOLDER_1 = "keypoints_new"
OUTPUT_FOLDER_2 = "original_kp_csv"
OUTPUT_FOLDER_3 = "new_kp_csv"

if not os.path.exists(OUTPUT_FOLDER_1):
    os.makedirs(OUTPUT_FOLDER_1)

if not os.path.exists(OUTPUT_FOLDER_2):
    os.makedirs(OUTPUT_FOLDER_2)

if not os.path.exists(OUTPUT_FOLDER_3):
    os.makedirs(OUTPUT_FOLDER_3)


def detect_missing_format(keypoints):
    """
    Detect how missing keypoints are represented.
    Returns a string: 'zero', 'zero_full', 'nan', 'neg', or 'none'
    NOTE: negative check only inspects X and Y channels (not Z),
    because MediaPipe z can be negative as relative depth.
    """
    if np.isnan(keypoints).any():
        return 'nan'

    # Completely empty array
    if keypoints.size == 0:
        return 'empty'

    # Zero check: if entire array is zeros
    if np.all(keypoints == 0):
        return 'zero_full'

    # Mixed zero check: if some keypoints are [0,0,0] or flattened zero rows
    if keypoints.ndim == 3:
        if (keypoints == 0).all(axis=-1).any():
            return 'zero'
    elif keypoints.ndim == 2:
        # If flattened (F, K*3), check rows of 3
        if (keypoints.reshape(keypoints.shape[0], -1, 3) == 0).all(axis=-1).any():
            return 'zero'

    # Negative check: consider only X and Y dims (indices 0 and 1)
    # Works after reshaping (we assume caller will reshape when needed)
    if keypoints.ndim == 3:
        xy = keypoints[:, :, :2]  # (F, K, 2)
        if (xy < 0).any():
            return 'neg'
    elif keypoints.ndim == 2 and keypoints.shape[1] % 3 == 0:
        # flattened: shape (F, K*3) -> reshape to (F,K,3) for checking
        kp3 = keypoints.reshape(keypoints.shape[0], keypoints.shape[1] // 3, 3)
        xy = kp3[:, :, :2]
        if (xy < 0).any():
            return 'neg'

    return 'none'


def reshape_keypoints_if_needed(keypoints):
    """
    Detect if keypoints are shape (F, 63) and reshape to (F, 21, 3).
    Works for any number of keypoints divisible by 3.
    """
    if keypoints.size == 0:
        raise ValueError("EMPTY .NPY FILE — keypoints not extracted properly")

    if keypoints.ndim == 2 and keypoints.shape[1] % 3 == 0:
        num_frames = keypoints.shape[0]
        num_kp = keypoints.shape[1] // 3
        return keypoints.reshape(num_frames, num_kp, 3)

    elif keypoints.ndim == 3:
        return keypoints  # already correct

    else:
        raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")


def convert_missing_to_nan(keypoints, missing_format):
    """
    Convert whatever missing format → np.nan
    NOTE: For 'neg' we only treat negative X/Y as missing (Z may be negative and is preserved).
    """
    kp = keypoints.copy()

    if missing_format == 'nan':
        return kp

    if missing_format in ('zero', 'zero_full'):
        # mark frames (or keypoints) that are all-zero as missing
        mask = (kp == 0).all(axis=-1)  # shape (frames, keypoints)
        kp[mask] = np.nan

    if missing_format == 'neg':
        # create mask for negative X or Y only
        # kp shape assumed (F, K, 3)
        xy = kp[:, :, :2]
        mask = (xy < 0).any(axis=-1)  # True where either x<0 or y<0
        # set only the x,y components to NaN and optionally z too (we set entire landmark to NaN)
        kp[mask] = np.nan

    return kp


def interpolate_keypoints(keypoints):
    num_frames, num_kp, dims = keypoints.shape

    for k in range(num_kp):
        for d in range(dims):
            series = pd.Series(keypoints[:, k, d])
            series_interp = series.interpolate(method='linear', limit_direction='both')
            keypoints[:, k, d] = series_interp.values

    return keypoints


def normalize_with_bbox(keypoints):
    num_frames, num_kp, dims = keypoints.shape
    kp_norm = np.zeros_like(keypoints)

    for i in range(num_frames):
        frame = keypoints[i]

        xs = frame[:, 0]
        ys = frame[:, 1]

        # bounding box
        x_min, x_max = np.nanmin(xs), np.nanmax(xs)
        y_min, y_max = np.nanmin(ys), np.nanmax(ys)

        width = max(x_max - x_min, 1e-6)
        height = max(y_max - y_min, 1e-6)

        kp_norm[i, :, 0] = (xs - x_min) / width
        kp_norm[i, :, 1] = (ys - y_min) / height

        if dims == 3:
            kp_norm[i, :, 2] = frame[:, 2]  # keep depth unchanged

    return kp_norm


def process_keypoint_file(npy_path):
    keypoints = np.load(npy_path)

    # If flattened (F, K*3), reshape to (F,K,3)
    keypoints = reshape_keypoints_if_needed(keypoints)

    missing_format = detect_missing_format(keypoints)
    print(f"{npy_path}  →  Missing format: {missing_format}")

    keypoints = convert_missing_to_nan(keypoints, missing_format)
    keypoints = interpolate_keypoints(keypoints)
    keypoints = normalize_with_bbox(keypoints)

    return keypoints


# -------------------
# Main loop
# -------------------
for kp_file in glob.glob(f"{INPUT_FOLDER}/*.npy"):
    try:
        original = np.load(kp_file)
        # Save original CSV safely (handle 1D/empty)
        if original.size == 0:
            print("Skipping empty file:", kp_file)
            continue

        # reshape original for CSV if needed
        if original.ndim == 2 and original.shape[1] % 3 == 0:
            orig_resh = original.reshape(original.shape[0], original.shape[1])
        elif original.ndim == 3:
            orig_resh = original.reshape(original.shape[0], -1)
        else:
            # fallback: try to reshape to (F, -1) if possible, else skip CSV
            try:
                orig_resh = original.reshape(original.shape[0], -1)
            except Exception:
                orig_resh = None

        processed = process_keypoint_file(kp_file)

        kp_name = os.path.splitext(os.path.basename(kp_file))[0]

        # Save PROCESSED for training
        np.save(os.path.join(OUTPUT_FOLDER_1, f"{kp_name}_processed.npy"), processed)

        # Save CSV for human-readable inspection (processed)
        df = pd.DataFrame(processed.reshape(processed.shape[0], -1))
        df.to_csv(os.path.join(OUTPUT_FOLDER_3, f"{kp_name}_processed.csv"), index=False)

        # Save ORIGINAL in CSV if possible
        if orig_resh is not None:
            df_orig = pd.DataFrame(orig_resh)
            df_orig.to_csv(os.path.join(OUTPUT_FOLDER_1, f"{kp_name}_original.csv"), index=False)

        print(f"Saved: {kp_name}_processed.npy + {kp_name}_processed.csv (+ original csv if available)")

    except Exception as e:
        print("ERROR processing", kp_file, ":", str(e))
