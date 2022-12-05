from PyQt5 import QtCore, QtWidgets, QtGui

from annotation.actions.Action import SideVolumeSplineResetAction, TiltedPlanesAnnotationAction, \
    DefaultPlanesAnnotationAction
from annotation.screens.Screen import Screen
from annotation.controlpanels.AnnotationControlPanel import AnnotationControlPanel
from annotation.visualization.panorex import CanvasPanorex
from annotation.visualization.sidevolume import CanvasSideVolume

class AnnotationScreen(Screen):
    def __init__(self, parent):
        super().__init__(parent)
        self.container.loaded.connect(self.show_)
        self.current_pos = 0

        self.layout.setAlignment(QtCore.Qt.AlignCenter)

        # panorex
        self.panorex = CanvasPanorex(self)
        self.panorex.spline_changed.connect(self._sidevolume_show)
        self.layout.addWidget(self.panorex, 0, 0)

        # side volume
        self.sidevolume = CanvasSideVolume(self)
        self.layout.addWidget(self.sidevolume, 0, 1, 2, 1)

        # non-zoomable sidevolume
        self.non_zoomable_sidevolume = CanvasSideVolume(self)
        self.layout.addWidget(self.non_zoomable_sidevolume, 0, 2, 2, 1)

        # control panel
        self.panel = AnnotationControlPanel()
        self.panel.pos_changed.connect(self.pos_changed_handler)
        self.panel.flags_changed.connect(self._sidevolume_show)
        self.panel.reset_annotation_clicked.connect(self.reset_annotation_clicked_handler)
        self.panel.acquire_annotation_clicked.connect(self.acquire_annotation_clicked_handler)
        self.panel.acquire_annotation_from_prediction_clicked.connect(self.acquire_annotation_from_prediction_clicked_handler)
        self.layout.addWidget(self.panel, 1, 0)

        self.right_arrow = QtWidgets.QShortcut(QtGui.QKeySequence.MoveToNextChar, self)
        self.left_arrow = QtWidgets.QShortcut(QtGui.QKeySequence.MoveToPreviousChar, self)
        self.up_arrow = QtWidgets.QShortcut(QtGui.QKeySequence.MoveToPreviousLine, self)
        self.down_arrow = QtWidgets.QShortcut(QtGui.QKeySequence.MoveToNextLine, self)
        self.zoom_in = QtWidgets.QShortcut(QtGui.QKeySequence('+'), self)
        self.zoom_out = QtWidgets.QShortcut(QtGui.QKeySequence('-'), self)

        self.right_arrow.activated.connect(lambda : self.change_zoom_pos(dx=+1))
        self.left_arrow.activated.connect(lambda : self.change_zoom_pos(dx=-1))
        self.up_arrow.activated.connect(lambda : self.change_zoom_pos(dy=-1))
        self.down_arrow.activated.connect(lambda : self.change_zoom_pos(dy=+1))
        self.zoom_in.activated.connect(lambda: self.zoom_in_())
        self.zoom_out.activated.connect(lambda: self.zoom_out_())

        self.sidevolume.paintEvent = self.update_both(self.sidevolume.paintEvent)
        self.non_zoomable_sidevolume.paintEvent = self.update_both(self.non_zoomable_sidevolume.paintEvent)

        #self.panel.setChildrenFocusPolicy(QtCore.Qt.NoFocus)

    def initialize(self):
        def yes(self):
            self.arch_handler.compute_side_volume(self.arch_handler.SIDE_VOLUME_SCALE, tilted=True)
            if not self.arch_handler.side_volume.correct:
                no(self)
            else:
                self.arch_handler.history.add(TiltedPlanesAnnotationAction())

        def no(self):
            self.arch_handler.compute_side_volume(self.arch_handler.SIDE_VOLUME_SCALE, tilted=False)
            self.arch_handler.history.add(DefaultPlanesAnnotationAction())

        self.arch_handler.save_state()
        self.mb.enable_save_load(True)
        self.mb.enable_(self.mb.annotation)
        self.arch_handler.offset_arch(pano_offset=0)
        title = "Tilted planes"
        if not self.arch_handler.L_canal_spline.is_empty() or not self.arch_handler.R_canal_spline.is_empty():
            message = "Would you like to use planes orthogonal to the IAN canal as base for the annotations?"
            self.messenger.question(title=title, message=message, yes=lambda: yes(self),
                                    no=lambda: no(self), default="no")
        else:
            message = "You will annotate on vertical slices because there are no canal splines."
            self.messenger.message(kind="information", title=title, message=message)
            no(self)

        self.panorex.set_img()
        self.panorex.set_can_edit_spline(not self.arch_handler.tilted())
        self.sidevolume.set_img()

    def pos_changed_handler(self):
        self.arch_handler.history.save_()
        self.arch_handler.annotation_masks.save_mask_splines()
        self.current_pos = self.panel.getPosValue()
        self.show_()

    def reset_annotation_clicked_handler(self):
        self.panel.auto_acquire_annotation.setChecked(False)
        self.arch_handler.annotation_masks.set_mask_spline(self.current_pos, None)
        self.arch_handler.history.add(SideVolumeSplineResetAction(self.current_pos))
        self._sidevolume_show()

    def acquire_annotation_clicked_handler(self):
        self.arch_handler.annotation_masks.get_mask_spline(self.current_pos, from_snake=True)
        self._sidevolume_show()

    def acquire_annotation_from_prediction_clicked_handler(self):
        self.arch_handler.annotation_masks.get_mask_spline(self.current_pos, from_prediction=True)
        self._sidevolume_show()

    def show_(self):
        self.panel.setPosSliderMaximum(len(self.arch_handler.arch.get_arch()) - 1)
        self.panel.setStep(self.arch_handler.annotation_masks.skip + 1)
        self.panorex.show_(pos=self.current_pos)
        self._sidevolume_show()

    def _sidevolume_show(self):
        self.sidevolume.show_(pos=self.current_pos,
                              show_dot=self.panel.show_dot.isChecked(),
                              auto_propagate=self.panel.auto_acquire_annotation.isChecked(),
                              show_mask_spline=self.panel.show_mask_spline.isChecked(),
                              show_cp_boxes=self.panel.show_cp_boxes.isChecked(),
                              normalize_mouse_hover=self.panel.normalize_mouse_hover.isChecked(),
                              show_network_prediction=self.panel.show_network_prediction.isChecked()
                              )
        self.non_zoomable_sidevolume.show_(pos=self.current_pos,
                              show_dot=self.panel.show_dot.isChecked(),
                              auto_propagate=self.panel.auto_acquire_annotation.isChecked(),
                              show_mask_spline=self.panel.show_mask_spline.isChecked(),
                              show_cp_boxes=False,
                              normalize_mouse_hover=self.panel.normalize_mouse_hover.isChecked(),
                              show_network_prediction=self.panel.show_network_prediction.isChecked()
                              )

    def connect_signals(self):
        pass

    def next_screen(self):
        pass

    def zoom_in_(self):
        self.sidevolume.zoom = min(self.sidevolume.zoom + 1, 8)
        self.show_()

    def zoom_out_(self):
        self.sidevolume.zoom = max(1, self.sidevolume.zoom - 1)
        if self.sidevolume.zoom == 1:
            self.sidevolume.zoom_pos = [0, 0]
        self.show_()

    def change_zoom_pos(self, dx=0, dy=0):
        if self.sidevolume.zoom == 1:
            return
        dx, dy = dx * 5, dy * 5
        x, y = self.sidevolume.zoom_pos
        self.sidevolume.set_zoom_pos(x + dx, y + dy)
        self.show_()
    def update_both(self, f):
        def g(e):
            f(e)
            self.sidevolume.update()
            self.non_zoomable_sidevolume.update()
        return g