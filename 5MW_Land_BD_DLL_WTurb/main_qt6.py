# main_ui.py
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from WT_qt6_1 import TuningUI  # 假设你的 TuningUI 在 tuning_ui.py 文件中
from WT_qt6_2 import ProcessTest       # 之前写的QProcess测试界面
from WT_qt6_3 import FastDataViewer

class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenFAST 全功能界面")
        self.resize(1200, 800)
        self._setup_layout()

    def _setup_layout(self):
        layout = QGridLayout()

        # 左上角 放 TuningUI
        tuning_widget = TuningUI()
        tuning_widget.setMinimumSize(300, 200)
        tuning_widget.setMaximumSize(600, 400)
        layout.addWidget(tuning_widget, 0, 0)

        # 左下角 放 ProcessTest
        process_widget = ProcessTest()
        process_widget.setMinimumSize(300, 200)
        process_widget.setMaximumSize(600, 400)
        layout.addWidget(process_widget, 1, 0)

        # 右上角 放 FastDataViewer（图形显示）
        fast_viewer = FastDataViewer()
        process_widget.setMinimumSize(300, 200)
        process_widget.setMaximumSize(600, 400)
        layout.addWidget(fast_viewer, 0, 1, 2, 1)  # 跨两行显示

        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec())










