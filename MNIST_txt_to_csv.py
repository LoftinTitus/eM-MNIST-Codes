import re
import csv
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

def parse_txt_to_csv(txt_path, csv_path):
    # Regex to match image number and corners
    img_num_re = re.compile(r'^Image\s*(\d+)')
    corners_re = re.compile(r'\[(\(\d+,\s*\d+\),\s*\(\d+,\s*\d+\),\s*\(\d+,\s*\d+\),\s*\(\d+,\s*\d+\))\]')
    results = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    image_number = None
    for line in lines:
        img_match = img_num_re.search(line)
        if img_match:
            image_number = img_match.group(1)
        corners_match = corners_re.search(line)
        if corners_match and image_number:
            corners_str = corners_match.group(1)
            corners = re.findall(r'\((\d+),\s*(\d+)\)', corners_str)
            if len(corners) == 4:
                results.append([image_number] + [coord for pair in corners for coord in pair])
            image_number = None  # Reset for next image
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_number', 'corner_1_x', 'corner_1_y', 'corner_2_x', 'corner_2_y', 'corner_3_x', 'corner_3_y', 'corner_4_x', 'corner_4_y'])
        writer.writerows(results)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle('TXT to CSV Converter')
    win.show()
    txt_path, _ = QFileDialog.getOpenFileName(win, 'Select .txt file', '', 'Text Files (*.txt)')
    if not txt_path:
        print('No .txt file selected.')
        sys.exit()
    csv_path, _ = QFileDialog.getSaveFileName(win, 'Save as .csv file', '', 'CSV Files (*.csv)')
    if not csv_path:
        print('No .csv file selected.')
        sys.exit()
    parse_txt_to_csv(txt_path, csv_path)
    print('CSV file created:', csv_path)