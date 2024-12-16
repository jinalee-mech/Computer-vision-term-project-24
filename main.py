from code.video_app import VideoApp
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoApp()
    ex.show()
    sys.exit(app.exec_())