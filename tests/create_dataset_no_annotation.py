import numpy as np
from conf import conf
import processing
from dataloader import load_dicom
import viewer


def no_annotation():
    """
    create dataset without annotation, the ground truth mask is a volume filled with zeros
    """

    # loading the data
    metadata, volume = load_dicom(conf.DICOM_DIR)

    # Z-axis has to be flipped
    volume = np.flip(volume, 0)

    # remove peak values
    volume = processing.quantiles(volume, min=0, max=0.985)

    # normalize volume between 0 and 1
    volume = processing.simple_normalization(volume)

    # choosing a slice and execute dental arch detection
    # speranzoni paziente 7 -> 98 - 169
    # paziente 1 -> 77 - 138
    # paziente 3 -> 85 - 169
    # paziente 8 -> 98 - 169

    section = volume[volume.shape[0] // 2 + 15]
    p, start, end = processing.arch_detection(section, debug=False)

    offset = 50

    l_offset, coords, h_offset, derivative = processing.arch_lines(p, start, end, offset=offset)
    # generating orthogonal lines to the offsets curves
    side_coords = processing.generate_side_coords(h_offset, l_offset, derivative, offset=2*offset)
    # volume of sections of the orthogonal lines
    side_volume = processing.canal_slice(volume, side_coords)
    # empty annotation volume as we have not annotation
    gt_side_volume = np.zeros_like(side_volume)

    # gif generation
    fake_gt = np.tile(gt_side_volume, (3, 1, 1, 1))  # overlay on the original image (colorful)
    fake_gt = np.moveaxis(fake_gt, 0, -1)
    processing.recap_on_gif(coords, h_offset, l_offset, side_volume, side_coords, section, fake_gt)

    # shuffling the slices before saving
    idx = np.random.rand(side_volume.shape[0]).argsort()
    side_volume = side_volume[idx]
    gt_side_volume = gt_side_volume[idx]

    if np.any(np.isnan(side_volume)) or np.any(np.isnan(gt_side_volume)):
        raise Exception('some resulting values are nan!')

    np.save(r'Y:\work\datasets\canal_segmentation\patient2\slices\idx.npy', idx)
    np.save(r'Y:\work\datasets\canal_segmentation\patient2\slices\data.npy', side_volume)
    np.save(r'Y:\work\datasets\canal_segmentation\patient2\slices\gt.npy', gt_side_volume)


if __name__ == "__main__":
    no_annotation()
