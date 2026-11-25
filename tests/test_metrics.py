import cv2
import numpy as np
import pytest

from evaluate import aggregate, cl_dice, compute_all, dice_score, eval_from_folders, iou_score, pixel_accuracy, precision_recall


def blank(size=64):
    return np.zeros((size, size), dtype=np.uint8)


def vessel(thickness=3, size=64):
    mask = blank(size)
    cv2.line(mask, (5, size // 2), (size - 6, size // 2), 255, thickness)
    return mask


def test_identical_masks_score_one():
    mask = vessel()
    metrics = compute_all(mask, mask)
    for name, value in metrics.items():
        assert value == pytest.approx(1.0), name


def test_disjoint_masks_score_zero():
    pred, gt = blank(), blank()
    pred[:10, :10] = 255
    gt[-10:, -10:] = 255
    assert dice_score(pred, gt) == pytest.approx(0.0, abs=1e-6)
    assert iou_score(pred, gt) == pytest.approx(0.0, abs=1e-6)
    assert cl_dice(pred, gt) == pytest.approx(0.0, abs=1e-6)
    precision, recall = precision_recall(pred, gt)
    assert precision == pytest.approx(0.0, abs=1e-6)
    assert recall == pytest.approx(0.0, abs=1e-6)


def test_half_overlap_matches_hand_calculation():
    pred, gt = blank(10), blank(10)
    pred[:, :6] = 255
    gt[:, 4:] = 255
    assert dice_score(pred, gt) == pytest.approx(2 * 20 / (60 + 60))
    assert iou_score(pred, gt) == pytest.approx(20 / 100)
    assert pixel_accuracy(pred, gt) == pytest.approx(20 / 100)
    assert precision_recall(pred, gt) == pytest.approx((20 / 60, 20 / 60))


def test_dice_and_iou_relation():
    pred, gt = vessel(3), vessel(7)
    dice, iou = dice_score(pred, gt), iou_score(pred, gt)
    assert dice == pytest.approx(2 * iou / (1 + iou), rel=1e-4)


def test_precision_and_recall_detect_error_direction():
    gt = vessel(5)
    under, over = vessel(1), vessel(15)
    under_precision, under_recall = precision_recall(under, gt)
    over_precision, over_recall = precision_recall(over, gt)
    assert under_precision == pytest.approx(1.0) and under_recall < 0.5
    assert over_recall == pytest.approx(1.0) and over_precision < 0.5


def test_empty_masks():
    empty = blank()
    assert dice_score(empty, empty) == pytest.approx(1.0)
    assert cl_dice(empty, empty) == 1.0
    assert cl_dice(vessel(), empty) == 0.0
    assert cl_dice(empty, vessel()) == 0.0
    assert dice_score(empty, vessel()) == pytest.approx(0.0, abs=1e-6)


def test_binarization_threshold():
    gt = vessel()
    noisy = np.where(gt > 0, 200, 100).astype(np.uint8)
    assert dice_score(noisy, gt) == pytest.approx(1.0)
    assert pixel_accuracy(noisy, gt) == 1.0


def test_cl_dice_ignores_vessel_thickness():
    gt, thick = vessel(3), vessel(9)
    assert dice_score(thick, gt) < 0.6
    assert cl_dice(thick, gt) == pytest.approx(1.0, abs=1e-3)


def test_cl_dice_penalizes_broken_vessel():
    gt = vessel(5)
    broken = gt.copy()
    broken[:, 20:44] = 0
    assert cl_dice(broken, gt) < dice_score(gt, gt)
    assert cl_dice(broken, gt) < 0.8


def test_aggregate_statistics():
    result = aggregate({"dice": [0.5, 0.7, 0.9]})
    assert result["dice"]["mean"] == pytest.approx(0.7)
    assert result["dice"]["std"] == pytest.approx(np.std([0.5, 0.7, 0.9]))
    assert (result["dice"]["min"], result["dice"]["max"]) == (0.5, 0.9)


def test_eval_from_folders(tmp_path):
    pred_dir, mask_dir = tmp_path / "pred", tmp_path / "mask"
    pred_dir.mkdir()
    mask_dir.mkdir()
    cv2.imwrite(str(pred_dir / "a.png"), vessel(3))
    cv2.imwrite(str(mask_dir / "a.png"), vessel(3))
    cv2.imwrite(str(pred_dir / "b.png"), blank())
    cv2.imwrite(str(mask_dir / "b.png"), vessel(3))
    cv2.imwrite(str(pred_dir / "orphan.png"), vessel(3))
    result = eval_from_folders(pred_dir, mask_dir)
    assert result["dice"]["max"] == pytest.approx(1.0)
    assert result["dice"]["min"] == pytest.approx(0.0, abs=1e-6)
    assert result["dice"]["mean"] == pytest.approx(0.5, abs=1e-6)


def test_eval_from_folders_requires_predictions(tmp_path):
    with pytest.raises(FileNotFoundError):
        eval_from_folders(tmp_path, tmp_path)
