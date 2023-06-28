from PyQt5 import QtWidgets, QtCore, QtGui
from annotation.controlpanels.SkipControlPanel import SkipControlPanel
from annotation.screens.AnnotationScreen import AnnotationScreen
from annotation.screens.Screen import Screen
from annotation.visualization.archview import SplineArchView
from annotation.visualization.panorex import CanvasPanorex


class PanorexSplineScreen(Screen):
    panorex_spline_selected = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.container.loaded.connect(self.show_)
        self.current_pos = 0

        # arch view
        self.archview = SplineArchView(self, show_LH_arch=False)
        self.archview.spline_changed.connect(self.spline_changed)
        self.layout.addWidget(self.archview, 0, 0)

        # panorex
        self.panorex = CanvasPanorex(self)
        self.layout.addWidget(self.panorex, 1, 0)

        # sparsity selector
        self.panel = SkipControlPanel()
        self.panel.skip_changed.connect(self.skip_changed_handler)
        self.layout.addWidget(self.panel, 2, 0)

        # continue button
        self.confirm_button = QtWidgets.QPushButton(self, text="Confirm (C)")
        self.confirm_button.setShortcut("C")
        self.confirm_button.clicked.connect(self.panorex_spline_selected.emit)
        self.layout.addWidget(self.confirm_button, 3, 0)

    def initialize(self):
        self.mb.enable_save_load(True)
        self.arch_handler.offset_arch(pano_offset=0)
        self.panorex.set_img()
        # self.panorex.set_can_edit_spline(self.arch_handler.gt_extracted)
        self.panorex.set_can_edit_spline(True)
        max_ = len(self.arch_handler.arch.get_arch()) - 1

        # It shouldn't be possible to skip more than 4 images
        self.panel.setNImages(max_)
        self.panel.setSkipMaximum(4)
        self.panel.setSkipValue(self.arch_handler.annotation_masks.skip)

    def spline_changed(self):
        self.arch_handler.update_coords()
        self.arch_handler.compute_side_coords()
        self.arch_handler.offset_arch()
        try:
            self.arch_handler.update_splines()
        except ValueError as e:
            print(e)
        self.show_()

    def skip_changed_handler(self, skip):
        self.arch_handler.annotation_masks.set_skip(skip)

    def show_(self):
        self.archview.show_(self.arch_handler.selected_slice, True)
        self.panorex.show_()

    def connect_signals(self):
        self.panorex_spline_selected.connect(self.next_screen)

    def next_screen(self):
        self.container.transition_to(AnnotationScreen)
