import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect

class CropWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Cropper')
        self.image = None
        self.crop_rect = QRect()
        self.cropping = False
        self.start_point = None
        self.end_point = None
        self.initUI()

    def initUI(self):
        self.label = QLabel('Open an image to crop.', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.open_btn = QPushButton('Open Image', self)
        self.open_btn.clicked.connect(self.open_image)
        self.save_btn = QPushButton('Save Crop', self)
        self.save_btn.clicked.connect(self.save_crop)
        self.save_btn.setEnabled(False)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.save_btn)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image files (*.jpg *.png *.bmp)')
        if fname:
            self.image = cv2.imread(fname)
            self.img_path = fname
            h, w, ch = self.image.shape
            max_dim = 800
            scale = min(max_dim / w, max_dim / h, 1.0)
            self.display_scale = scale
            if scale < 1.0:
                disp_img = cv2.resize(self.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                disp_h, disp_w = disp_img.shape[:2]
            else:
                disp_img = self.image.copy()
                disp_h, disp_w = h, w
            bytes_per_line = disp_img.shape[2] * disp_w
            cv_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            qimg = QImage(cv_img.data, disp_w, disp_h, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(self.pixmap)
            self.label.setFixedSize(disp_w, disp_h)
            self.save_btn.setEnabled(True)
            self.label.mousePressEvent = self.mouse_press
            self.label.mouseMoveEvent = self.mouse_move
            self.label.mouseReleaseEvent = self.mouse_release

    def mouse_press(self, event):
        if self.image is not None:
            self.cropping = True
            self.start_point = event.pos()
            self.end_point = event.pos()

    def mouse_move(self, event):
        if self.cropping:
            self.end_point = event.pos()
            self.update_crop_rect()

    def mouse_release(self, event):
        if self.cropping:
            self.end_point = event.pos()
            self.cropping = False
            self.update_crop_rect()
            self.show_crop()

    def update_crop_rect(self):
        x1, y1 = self.start_point.x(), self.start_point.y()
        x2, y2 = self.end_point.x(), self.end_point.y()
        self.crop_rect = QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        temp_pixmap = QPixmap(self.pixmap)
        painter = QPainter(temp_pixmap)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        painter.drawRect(self.crop_rect)
        painter.end()
        self.label.setPixmap(temp_pixmap)

    def show_crop(self):
        if self.crop_rect.width() > 0 and self.crop_rect.height() > 0:
            x, y, w, h = self.crop_rect.x(), self.crop_rect.y(), self.crop_rect.width(), self.crop_rect.height()
            # Map display coordinates back to original image
            scale = getattr(self, 'display_scale', 1.0)
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(w / scale)
            orig_h = int(h / scale)
            cropped = self.image[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
            # Scale cropped image for preview
            preview_max_dim = 400
            ch2 = cropped.shape[2] if len(cropped.shape) == 3 else 1
            h2, w2 = cropped.shape[:2]
            preview_scale = min(preview_max_dim / w2, preview_max_dim / h2, 1.0)
            if preview_scale < 1.0:
                preview_img = cv2.resize(cropped, (int(w2 * preview_scale), int(h2 * preview_scale)), interpolation=cv2.INTER_AREA)
                preview_h, preview_w = preview_img.shape[:2]
            else:
                preview_img = cropped.copy()
                preview_h, preview_w = h2, w2
            cv_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            bytes_per_line2 = ch2 * preview_w
            qimg2 = QImage(cv_img.data, preview_w, preview_h, bytes_per_line2, QImage.Format_RGB888)
            crop_win = QLabel()
            crop_win.setPixmap(QPixmap.fromImage(qimg2))
            crop_win.setWindowTitle('Cropped Image')
            crop_win.setFixedSize(preview_w, preview_h)
            crop_win.show()
            self.cropped_img = cropped
            self.corners = [
                (orig_x, orig_y),
                (orig_x + orig_w, orig_y),
                (orig_x + orig_w, orig_y + orig_h),
                (orig_x, orig_y + orig_h)
            ]
            print('Corners relative to original image:', self.corners)
            self.crop_win = crop_win

    def save_crop(self):
        if hasattr(self, 'cropped_img'):
            # Use the native macOS Finder dialog for saving files
            fname, _ = QFileDialog.getSaveFileName(
                self,
                'Save Cropped Image',
                '',
                'Image files (*.jpg *.png *.bmp)'
            )
            if fname:
                cv2.imwrite(fname, self.cropped_img)
                print('Cropped image saved:', fname)
                print('Corners:', self.corners)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CropWindow()
    win.show()
    sys.exit(app.exec_())