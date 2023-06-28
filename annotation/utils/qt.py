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
    red_square = None

    if mousePos is not None and mousePos != (0, 0):
        OFFSET = 13
        x, y = mousePos
        
        start_y = y-squareSize[0]-OFFSET
        end_y = y+squareSize[0]-OFFSET
        start_x = x-squareSize[1]-OFFSET
        end_x = x+squareSize[1]-OFFSET

        start_y = max(0, start_y)
        start_x = max(0, start_x)
        end_y = max(0, end_y)
        end_x = max(0, end_x)

        start_y = min(img_.shape[0] - 1, start_y)
        start_x = min(img_.shape[1] - 1, start_x)
        end_y = min(img_.shape[0] - 1, end_y)
        end_x = min(img_.shape[1] - 1, end_x)

        cs_area = img_[start_y:end_y, start_x:end_x]
        red_square = np.zeros((img_.shape[0], img_.shape[1]))
        red_square[start_y:end_y, start_x] = 1
        red_square[start_y:end_y, end_x] = 1
        red_square[start_y, start_x:end_x] = 1
        red_square[end_y, start_x:end_x] = 1

        if cs_area.shape[0] != 0 and cs_area.shape[1] != 0:
            area_max = np.max(cs_area)
            area_min = np.min(cs_area)
            if area_max > 0.75:
                area_max = 0.75
            if area_max - area_min != 0:
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

    if red_square is not None:
        img_[red_square == 1, :] = [255, 0, 0]

    #img_[start_x, start_y:end_y] = []

    h, w, c = img_.shape
    step = w * c
    # must pass a copy of the image
    img_ = img_.astype(np.uint8)
    qimage = QtGui.QImage(img_.copy(), w, h, step, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap(qimage)
    return pixmap
