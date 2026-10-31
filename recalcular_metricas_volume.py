"""Compare full-volume, pooled-patch, and per-patch Dice/IoU calculations."""

import csv
import struct
from pathlib import Path

import numpy as np

SAMPLES = ("03_09", "04_09", "06_06", "07_06", "07_09", "09_06", "11_06", "22_08", "23_08", "28_08", "29_08")
MODELS = ("resnet18", "inceptionv3", "resunet")
TAGS = {256, 257, 258, 259, 273, 277, 278, 279, 339}
SMOOTH = 1e-5  # Same smoothing constant as segmentation_models_3D.
PATCH_SIZE = 64


def read_stack(path):
    """Read the uncompressed, one-strip-per-slice TIFFs used by these runs."""
    with open(path, "rb") as f:
        header = f.read(8)
        endian = "<" if header[:2] == b"II" else ">" if header[:2] == b"MM" else None
        if endian is None or struct.unpack(endian + "H", header[2:4])[0] != 42:
            raise ValueError(f"Unsupported TIFF header: {path}")

        pages, offset = [], struct.unpack(endian + "I", header[4:8])[0]
        while offset:
            f.seek(offset)
            count = struct.unpack(endian + "H", f.read(2))[0]
            tags = {}
            for _ in range(count):
                entry = f.read(12)
                tag, kind, n = struct.unpack(endian + "HHI", entry[:8])
                if tag not in TAGS:
                    continue
                if kind not in (3, 4):  # TIFF SHORT or LONG
                    raise ValueError(f"Unexpected TIFF tag type {kind} in {path}")
                fmt, size = ("H", 2) if kind == 3 else ("I", 4)
                raw = entry[8:8 + n * size] if n * size <= 4 else None
                if raw is None:
                    f.seek(struct.unpack(endian + "I", entry[8:12])[0])
                    raw = f.read(n * size)
                tags[tag] = struct.unpack(endian + fmt * n, raw)
            offset = struct.unpack(endian + "I", f.read(4))[0]
            pages.append(tags)

        width, height = pages[0][256][0], pages[0][257][0]
        bits = pages[0][258][0]
        if bits not in (8, 16, 32, 64):
            raise ValueError(f"Unsupported pixel depth ({bits}) in {path}")
        sample_format = pages[0].get(339, (1,))[0]  # unsigned unless TIFF says signed
        dtype = np.dtype(endian + ("i" if sample_format == 2 else "u") + str(bits // 8))
        stack = np.empty((len(pages), height, width), dtype=dtype)

        for z, tags in enumerate(pages):
            if tags[259][0] != 1 or tags.get(277, (1,))[0] != 1:
                raise ValueError(f"Expected uncompressed, single-channel TIFF: {path}")
            if tags[278][0] != height or len(tags[273]) != 1:
                raise ValueError(f"Expected one full-image strip per TIFF page: {path}")
            f.seek(tags[273][0])
            raw = f.read(tags[279][0])
            stack[z] = np.frombuffer(raw, dtype=dtype).reshape(height, width)
    return stack


def score_counts(tp, fp, fn, smooth=SMOOTH):
    union = tp + fp + fn
    iou = (tp + smooth) / (union + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    return iou, dice


def counts_for_class(prediction, truth, class_id):
    pred_class = prediction == class_id
    true_class = (truth != 0) == bool(class_id)
    tp = int(np.count_nonzero(pred_class & true_class))
    fp = int(np.count_nonzero(pred_class & ~true_class))
    fn = int(np.count_nonzero(~pred_class & true_class))
    return tp, fp, fn


def split_into_patches(volume):
    z, y, x = volume.shape
    p = PATCH_SIZE
    if z % p or y % p or x % p:
        raise ValueError(f"Volume dimensions must be divisible by {p}: {volume.shape}")
    nz, ny, nx = z // p, y // p, x // p
    return volume.reshape(nz, p, ny, p, nx, p).transpose(0, 2, 4, 1, 3, 5).reshape(-1, p, p, p)


def calculate_aggregation_modes(prediction, reference):
    """Compare full-volume pooling with patch pooling and per-patch means."""
    if prediction.shape != reference.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape}, reference {reference.shape}")

    # Mode 1: count every voxel in the complete volume, then score each class.
    global_class_scores = {}
    for class_id in (0, 1):
        tp, fp, fn = counts_for_class(prediction, reference, class_id)
        global_class_scores[class_id] = score_counts(tp, fp, fn)
    volume_fg_iou, volume_fg_dice = global_class_scores[1]
    volume_macro_iou = np.mean([global_class_scores[c][0] for c in (0, 1)])
    volume_macro_dice = np.mean([global_class_scores[c][1] for c in (0, 1)])

    # Mode 2: sum counts from the 64 patches before scoring; this should equal mode 1.
    pred_patches = split_into_patches(prediction)
    true_patches = split_into_patches(reference)
    patch_class_counts = {0: [], 1: []}
    for p_pred, p_true in zip(pred_patches, true_patches):
        for class_id in (0, 1):
            patch_class_counts[class_id].append(counts_for_class(p_pred, p_true, class_id))
    pooled_scores = {}
    for class_id, counts in patch_class_counts.items():
        pooled = tuple(sum(c[i] for c in counts) for i in range(3))
        pooled_scores[class_id] = score_counts(*pooled)
    pooled_fg_iou, pooled_fg_dice = pooled_scores[1]
    pooled_macro_iou = np.mean([pooled_scores[c][0] for c in (0, 1)])
    pooled_macro_dice = np.mean([pooled_scores[c][1] for c in (0, 1)])

    # Mode 3: score each patch first, then average the 64 scores.
    patch_fg_scores, patch_macro_scores = [], []
    empty_foreground_patches = 0
    for p_pred, p_true in zip(pred_patches, true_patches):
        class_scores = {}
        for class_id in (0, 1):
            class_scores[class_id] = score_counts(*counts_for_class(p_pred, p_true, class_id))
        patch_fg_scores.append(class_scores[1])
        patch_macro_scores.append((
            np.mean([class_scores[c][0] for c in (0, 1)]),
            np.mean([class_scores[c][1] for c in (0, 1)]),
        ))
        if not np.any(p_true != 0):
            empty_foreground_patches += 1
    patch_fg_iou, patch_fg_dice = np.mean(patch_fg_scores, axis=0)
    patch_macro_iou, patch_macro_dice = np.mean(patch_macro_scores, axis=0)

    return {
        "volume_foreground_iou": volume_fg_iou,
        "volume_foreground_dice": volume_fg_dice,
        "pooled64_foreground_iou": pooled_fg_iou,
        "pooled64_foreground_dice": pooled_fg_dice,
        "mean64_foreground_iou": patch_fg_iou,
        "mean64_foreground_dice": patch_fg_dice,
        "volume_macro_iou": volume_macro_iou,
        "volume_macro_dice": volume_macro_dice,
        "pooled64_macro_iou": pooled_macro_iou,
        "pooled64_macro_dice": pooled_macro_dice,
        "mean64_macro_iou": patch_macro_iou,
        "mean64_macro_dice": patch_macro_dice,
        "patch_count": len(pred_patches),
        "empty_foreground_patches": empty_foreground_patches,
    }


def main():
    d = Path("/mnt/d/mestrado/notebooks")
    if not d.exists():
        d = Path("D:/mestrado/notebooks")
    source, masks = d / "LOO_3d_unet_backbones", d / "steel/stl"
    output = Path(__file__).resolve().parent
    rows = []

    for model in MODELS:
        for sample in SAMPLES:
            pred_path = source / model / "3D_result" / f"{sample}_segmented.tif"
            mask_path = masks / sample / "output_fibras.tif"
            pred, truth = read_stack(pred_path), read_stack(mask_path)
            if not np.isin(np.unique(pred), (0, 1)).all():
                raise ValueError(f"Expected saved argmax labels 0/1: {pred_path}")
            scores = calculate_aggregation_modes(pred, truth)
            rows.append({
                "model": model, "sample": sample,
                **scores,
                "voxels": pred.size, "volume_shape": "x".join(map(str, pred.shape)),
                "prediction_rule": "saved argmax class labels 0/1; original softmax probabilities unavailable",
                "reference_rule": "foreground is reference voxel > 0",
                "smooth": SMOOTH,
                "prediction_file": str(pred_path), "reference_file": str(mask_path),
            })
            print(f"{model:12s} {sample}: volume FG={scores['volume_foreground_iou']:.4f}/{scores['volume_foreground_dice']:.4f}; "
                  f"pooled-64 FG={scores['pooled64_foreground_iou']:.4f}/{scores['pooled64_foreground_dice']:.4f}; "
                  f"mean-64 FG={scores['mean64_foreground_iou']:.4f}/{scores['mean64_foreground_dice']:.4f}")

    detail_path = output / "metricas_volume_por_amostra.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        for metric in ("volume_foreground_iou", "volume_foreground_dice", "pooled64_foreground_iou", "pooled64_foreground_dice",
                       "mean64_foreground_iou", "mean64_foreground_dice", "volume_macro_iou", "volume_macro_dice",
                       "pooled64_macro_iou", "pooled64_macro_dice", "mean64_macro_iou", "mean64_macro_dice"):
            values = np.array([r[metric] for r in group])
            summary.append({
                "model": model, "metric": metric, "n_samples": len(values),
                "mean": values.mean(), "std_sample_ddof1": values.std(ddof=1),
                "std_population_ddof0": values.std(ddof=0),
            })
    summary_path = output / "resumo_volume_por_modelo.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSaved {detail_path}")
    print(f"Saved {summary_path}")
    for r in summary:
        print(f"{r['model']:12s} {r['metric']}: mean={r['mean']:.8f}, sample SD={r['std_sample_ddof1']:.8f}")


if __name__ == "__main__":
    main()
