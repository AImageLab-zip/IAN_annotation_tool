from PyQt5 import QtWidgets
from pyface.qt import QtGui

from annotation.components.Dialog import question
from annotation.components.Menu import Menu
from annotation.containers.MainContainer import Container


class Window(QtGui.QMainWindow):
    WINDOW_TITLE = "IAN Annotation Tool"

    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle(self.WINDOW_TITLE)

        self.menubar = Menu(self)
        self.menubar.open.connect(self.open_dicomdir)
        self.setMenuBar(self.menubar.get())

        self.container = Container(self)
        self.setCentralWidget(self.container)

    def open_dicomdir(self):
        dialog = QtWidgets.QFileDialog()
        file_path = dialog.getOpenFileName(None, "Select DICOMDIR file", filter="DICOMDIR")
        path = file_path[0]
        if path:
            self.setWindowTitle("{} - [{}]".format(self.WINDOW_TITLE, path))
            self.container.dicomdir_changed(path)

    def closeEvent(self, event):
        title = "Are you sure you want to quit?"
        message = "Unsaved changes will be lost."
        question(self, title, message, event.accept, event.ignore)
