from pyface.qt import QtGui
from annotation.widgets.panorex import SinglePanorexWidget
from annotation.widgets.sidevolume import SideVolume


class AnnotationContainerWidget(QtGui.QWidget):
    def __init__(self, parent):
        super(AnnotationContainerWidget, self).__init__()
        self.container = parent

        self.layout = QtGui.QGridLayout(self)

        # panorex
        self.panorex = SinglePanorexWidget(self)
        self.layout.addWidget(self.panorex, 0, 0)

        # side volume
        self.sidevolume = SideVolume(self)
        self.layout.addWidget(self.sidevolume, 0, 1)

        self.arch_handler = None
        self.current_pos = 0

    def show_img(self):
        self.panorex.show_panorex(pos=self.current_pos)
        self.sidevolume.show_side_view(pos=self.current_pos)

    def set_arch_handler(self, arch_handler):
        self.arch_handler = arch_handler
        self.panorex.arch_handler = arch_handler
        self.sidevolume.arch_handler = arch_handler
