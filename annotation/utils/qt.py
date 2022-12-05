import numpy as np
from pyface.qt import QtGui
from annotation.utils.ContrastStretching import ContrastStretching
import cv2

def numpy2pixmap(data, mousePos=None, squareSize=(30, 30)):
    """
    Converts a 2D/3D numpy array to a QPixmap

    Args:
        data (np.ndarray): 2D/3D image
        mousePos (tuple): (x, y) position of the mouse

    Returns:
        (pyface.qt.QtGui.QPixmap): pixmap of the image
    """
    # img_ = cv2.resize(img_, (img_.shape[0] * 5, img_.shape[1] * 5), interpolation=cv2.INTER_AREA)

    img_ = np.clip(data, 0, 1)
    cs = ContrastStretching()

    if mousePos is not None:
        OFFSET = 13
        x, y = mousePos
        
        start_y = y-squareSize[0]-OFFSET
        end_y = y+squareSize[0]-OFFSET
        start_x = x-squareSize[1]-OFFSET
        end_x = x+squareSize[1]-OFFSET

        if start_y < 0: start_y = 0
        if start_x < 0: start_x = 0
        if end_y < 0: end_y = 0
        if end_x < 0: end_x = 0

        cs_area = img_[start_y:end_y, start_x:end_x]

        if cs_area.shape[0] != 0 and cs_area.shape[1] != 0:
            area_max = np.max(cs_area)
            area_min = np.min(cs_area)
            if len(cs_area.shape) == 3:
                cs_area[:,:,0] = (cs_area[:,:,0] - cs_area[:,:,1])*0.2 + cs_area[:,:,1]
                area_max = np.max(cs_area[:,:,1])
                area_min = np.min(cs_area[:,:,1])
            # cs_area = cs_area > ((area_max+area_min)/2)
            cs_area = (cs_area - area_min)/(area_max - area_min)
            img_[start_y:end_y, start_x:end_x] = cs_area

    img_ = cs.stretch(img_)
    img_ = np.clip(img_ * 255, 0, 255)

    if len(img_.shape) != 3:
        img_ = np.stack((img_,) * 3, axis=-1)

    h, w, c = img_.shape
    step = w * c
    # must pass a copy of the image
    img_ = img_.astype(np.uint8)
    qimage = QtGui.QImage(img_.copy(), w, h, step, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap(qimage)
    return pixmap
