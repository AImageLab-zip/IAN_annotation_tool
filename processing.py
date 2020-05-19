import numpy as np
import viewer
import cv2
import imageio


def x_slice(volume, fixed_val):
    """
    create a lateral 2D projection for the
    :param volume: 3D numpy volume
    :param fixed_val: fixed coord over the X
    :return: 2D numpy image
    """
    return np.squeeze(volume[:, :, fixed_val])


def y_slice(volume, fixed_val):
    """
    create a front 2D projection for the
    :param volume:
    :param fixed_val:
    :return:
    """
    return np.squeeze(volume[:, fixed_val, :])


def quantiles(volume, min=0.02, max=0.98):
    min = np.quantile(volume, min),
    max = np.quantile(volume, max)
    volume[volume > max] = max
    volume[volume < min] = min
    return volume

def canal_slice(volume, coords, derivative):
    """
    :param volume:
    :param coords:
    :param derivative:
    :return:
    """

    # creating the set of perpendicular points from the curve
    slices = []
    for (x1, y1), (x2, y2), alfa in zip(coords[0], coords[1], derivative):
        sign = 1 if alfa > 0 else -1
        x_dist = 1 + int(np.ceil(abs(x1 - x2)))
        y_dist = 1 + int(np.ceil(abs(y1 - y2)))
        cos = np.sqrt(1/(alfa**2 + 1))
        sin = np.sqrt(alfa ** 2 / (alfa ** 2 + 1))
        points = [
            [x1 + sign * i * cos, np.floor(y1 + (i * sin))] for i in
                  range(max(x_dist, y_dist))
        ]
        slices.append(points)

    # creating volume from the points
    h = volume.shape[0]
    w = max([len(points) for points in slices])
    z = len(slices)
    if len(volume.shape) == 3:
        cut = np.zeros((z, h, w), np.float32)
        for z_id, points in enumerate(slices):
            for w_id, (x, y) in enumerate(points):
                cut[z_id, :, w_id] = volume[:, int(y), int(x)]
    elif len(volume.shape) == 4:
        cut = np.zeros((z, h, w, 3), np.float32)
        for z_id, points in enumerate(slices):
            for w_id, (x, y) in enumerate(points):
                cut[z_id, :, w_id, :] = volume[:, int(y), int(x), :]
    else:
        raise Exception("weird volume shape")

    return cut, slices


def recap_on_gif(coords, high_offset, low_offset, side_volume, side_coords, slice, gt_side_volume):
    """

    :param coords:
    :param high_offset:
    :param low_offset:
    :param side_volume:
    :param side_coords:
    :param slice:
    :param gt_side_volume:
    :return:
    """

    slice = cv2.normalize(slice, slice, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    original = np.tile(slice, (3, 1, 1))  # overlay on the original image (colorful)
    original = np.moveaxis(original, 0, -1)

    # drawing the line and the offsets of the upper view
    for idx in range(len(coords)):
        original[int(coords[idx][1]), int(coords[idx][0])] = (255, 0, 0)
        original[int(high_offset[idx][1]), int(high_offset[idx][0])] = (0, 255, 0)
        original[int(low_offset[idx][1]), int(low_offset[idx][0])] = (0, 255, 0)

    # create an upper view for each section
    sections = []
    for points in side_coords:
        tmp = original.copy()
        for x, y in points:
            tmp[int(y), int(x)] = (0, 0, 255)
        sections.append(tmp)
    sections = np.stack(sections)

    # rescaling the projection volume properly
    y_ratio = original.shape[0] / side_volume.shape[1]
    width = int(side_volume.shape[2] * y_ratio)
    height = int(side_volume.shape[1] * y_ratio)
    scaled_side_volume = np.ndarray(shape=(side_volume.shape[0], height, width))
    scaled_gt_volume = np.ndarray(shape=(gt_side_volume.shape[0], height, width, 3))
    for i in range(side_volume.shape[0]):
        scaled_side_volume[i] = cv2.resize(side_volume[i, :, :], (width, height), interpolation=cv2.INTER_AREA)
        scaled_gt_volume[i] = cv2.resize(gt_side_volume[i, :, :], (width, height), interpolation=cv2.INTER_AREA)

    # padding the side volume and rescaling
    # pad_side_volume = np.zeros((side_volume.shape[0], original.shape[0], original.shape[1]))
    # pad_side_volume[:, :side_volume.shape[1], :side_volume.shape[2]] = side_volume
    scaled_side_volume = cv2.normalize(scaled_side_volume, scaled_side_volume, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scaled_gt_volume = cv2.normalize(scaled_gt_volume, scaled_gt_volume, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # creating RGB volume
    scaled_side_volume = np.tile(scaled_side_volume, (3, 1, 1, 1))  # overlay on the original image (colorful)
    scaled_side_volume = np.moveaxis(scaled_side_volume, 0, -1)

    # GIF creation
    gif_source = np.concatenate((sections, scaled_side_volume, scaled_gt_volume), axis=2)
    gif = []
    for i in range(gif_source.shape[0]):
        gif.append(gif_source[i, :, :])
    imageio.mimsave('test.gif', gif)


def get_annotations(metadata):
    """
    return a mask which maps on each pixel of the input image a ground truth value 0 or 1
    :param metadata: metadata for a given slice
    :return: mask extracted from the metadata
    """
    '''
    # alternative method that does not work that good
    overlay_data = dcm[0x6004, 0x3000].value
    rows = dcm[0x6004, 0x0010].value
    cols = dcm[0x6004, 0x0011].value

    btmp = np.frombuffer(overlay_data, dtype=np.uint8)
    btmp = np.unpackbits(btmp)
    btmp = btmp[:rows * cols]
    btmp = np.reshape(btmp, (rows, cols))
    return btmp
    '''
    return metadata.overlay_array(0x6004)


def get_annotated_volume(metadata):
    """
    return a volume of masks according to the overlays fields
    :param metadata: volume of metadata
    :return: a binary volume
    """
    annotations = []
    for meta in metadata:
        annotations.append(get_annotations(meta))
    return np.stack(annotations).astype(np.bool_)


def simple_normalization(data):
    """
    normalize slice or volumes between 0 and 1
    :param data: numpy image or volume
    :return: numpy normalized image
    """
    return data.astype(np.float32)/data.max()


def simple_interpolation(x_func, y_func, volume):
    """
    simple interpolation between four pixels of the image given a float set of coords
    :param x_func: float x coord
    :param y_func: float y coord on the spline
    :param volume: 3D volume of the dental image
    :return: a numpy array over the Z axis of the volume on a fixed (x,y) obtained by interpolation
    """
    x1, x2 = int(np.ceil(x_func)), int(np.floor(x_func))
    y1, y2 = int(np.ceil(y_func)), int(np.floor(y_func))
    dx, dy = x_func - x2, y_func - y2
    P1 = volume[:, y1, x1] * (1 - dx) * (1 - dy)
    P2 = volume[:, y2, x1] * dx * (1 - dy)
    P3 = volume[:, y1, x2] * dx * dy
    P4 = volume[:, y2, x2] * (1 - dx) * dy
    return np.sum(np.stack((P1, P2, P3, P4)), axis=0)


def compute_skeleton(img):
    """
    create the skeleton using morphology
    :param img: source image
    :return: image with the same input shape, b&w: 0 background, 255 skeleton elements
    """
    img = img.astype(np.uint8)
    size = img.size
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = np.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            return skel


def arch_detection(slice, debug=False):
    """
    compute a polynomial spline of the dental arch from a DICOM file
    :param slice: the source image
    :return: a polinomial funtion, start and end X coords from there the dental arch can be tracked
    """
    if debug:
        viewer.plot_2D(slice)
    # initial closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    arch = cv2.morphologyEx(slice, cv2.MORPH_CLOSE, kernel)
    # simple threshold -> switch to uint8
    ret, arch = cv2.threshold(arch, 0.5, 1, cv2.THRESH_BINARY)
    arch = arch.astype(np.uint8)
    if debug:
        viewer.plot_2D(arch)

    # hole filling
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    arch = cv2.morphologyEx(arch, cv2.MORPH_CLOSE, kernel)
    if debug:
        viewer.plot_2D(arch)

    # major filtering with labelling
    ret, labels = cv2.connectedComponents(arch)
    for label in range(1, ret):
        if labels[labels == label].size < 10000:
            labels[labels == label] = 0
    if debug:
        viewer.plot_2D(labels)

    # compute skeleton
    skel = compute_skeleton(labels)
    if debug:
        viewer.plot_2D(skel)

    # labelling on the resulting skeleton
    cs, im = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for c in cs:
        if len(c) > 40:
            filtered.append(c)

    # creating the mask of chosen pixels
    contour = np.zeros(skel.shape, np.uint8)
    cv2.drawContours(contour, filtered, -1, 255)
    if debug:
        viewer.plot_2D(contour)

    # regression polynomial function
    coords = np.argwhere(contour > 0)
    y = [y for y, x in coords]
    x = [x for y, x in coords]
    pol = np.polyfit(x, y, 12)
    p = np.poly1d(pol)

    # generating the curve for a check
    if debug:
        recon = np.zeros(skel.shape, np.uint8)  # binary image for test
        original_rgb = np.tile(slice, (3, 1, 1))  # overlay on the original image (colorful)
        original_rgb = np.moveaxis(original_rgb, 0, -1)
        for sample in range(min(x), max(x)):
            y_sample = p(sample)
            recon[int(y_sample), sample] = 255
            original_rgb[int(y_sample), sample, :] = (255, 0, 0)
        viewer.plot_2D(recon)
        viewer.plot_2D(original_rgb, cmap=None)

    return p, min(x), max(x)


def arch_lines(func, start, end, offset=50):
    """

    :param func:
    :param start:
    :param end:
    :param offset:
    :return:
    """
    d = 1
    delta = 0.3
    coords = []
    x = start + 1
    # we start from the range of X values on the X axis,
    # we create a new list X of x coords along the curve
    # we exploit the first order derivative to place values in X
    # so that f(X) is equally distant for each point in X
    while x < end:
        coords.append((x, func(x)))
        alfa = (func(x+delta/2) - func(x-delta/2)) / delta
        x = x + d * np.sqrt(1/(alfa**2 + 1))

    # creating lines parallel to the spline
    high_offset = []
    low_offset = []
    derivative = []
    for x, y in coords:
        alfa = (func(x + delta / 2) - func(x - delta / 2)) / delta  # first derivative
        alfa = -1 / alfa  # perpendicular coeff
        cos = np.sqrt(1/(alfa**2 + 1))
        sin = np.sqrt(alfa ** 2 / (alfa ** 2 + 1))
        if alfa > 0:
            low_offset.append((x + offset * cos, y + offset * sin))
            high_offset.append((x - offset * cos, y - offset * sin))
        else:
            low_offset.append((x - offset * cos, y + offset * sin))
            high_offset.append((x + offset * cos, y - offset * sin))
        derivative.append(alfa)

    return low_offset, coords, high_offset, derivative


def create_panorex(volume, coords, high_offset, low_offset):
    """

    :param volume:
    :param coords:
    :param high_offset:
    :param low_offset:
    :return:
    """
    z_shape, y_shape, x_shape = volume.shape
    # better re-projection using bi-linear interpolation
    panorex = np.zeros((z_shape, len(coords)), np.float32)
    for idx, (x, y) in enumerate(coords):
        panorex[:, idx] = simple_interpolation(x, y, volume)

    # re-projection of the offsets curves
    panorex_up = np.zeros((z_shape, len(high_offset)), np.int)
    panorex_down = np.zeros((z_shape, len(low_offset)), np.int)
    for idx, (x, y) in enumerate(high_offset):
        panorex_up[:, idx] = simple_interpolation(x, y, volume)
    for idx, (x, y) in enumerate(low_offset):
        panorex_down[:, idx] = simple_interpolation(x, y, volume)

    viewer.plot_2D(panorex_down, cmap='bone')
    viewer.plot_2D(panorex, cmap='bone')
    viewer.plot_2D(panorex_up, cmap='bone')
    return panorex