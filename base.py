# dùng streamlit tạo giao diện web nhận diện biển số nhận vào 1 ảnh và trả về ảnh đã nhận diện
# ảnh đã nhận diện sẽ được lưu vào thư mục static
# cài đặt streamlit: pip install streamlit
# chạy lệnh: streamlit run base.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import math

from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras import layers

# Khai báo các biến cần thiết
text_char = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
             13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'K', 19: 'L', 20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S',
             25: 'T', 26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}

# Hàm load model nhận diện ký tự
def load_character_recognition_model(weights_path):
    """Load the character recognition model."""
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(31, activation='softmax'))
    model.load_weights(weights_path)
    return model

# Hàm xoay ảnh
def rotate_image(image, center, angle, points=None):
    """Rotate the image around the center and return the rotated image along with rotated points if provided."""
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if points is not None:
        new_points = cv2.transform(np.array([points]), M)[0]
        return rotated_image, new_points
    return rotated_image

# Load model nhận diện biển số và ký tự
license_plate_detector = YOLO("./models/yolov8_obb_detect_license.pt")
number_detector = YOLO("./models/character_yolov8_v2.pt")
model = load_character_recognition_model("./models/CNN_number_recognition.h5")

# Hàm phát hiện biển số
def detect_plate_normal(image):
    results = license_plate_detector(image)[0]
    for result in results:
        x1, y1, x2, y2 = result.obb.xyxy.numpy()[0]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image

# Hàm phát hiện biển số và cắt ảnh biển số
def detect_plate(image):
    license_plates = license_plate_detector(image)[0]
    if license_plates.obb.xywhr.shape[0] == 0:
        print("No license plate detected.")
        return image
    for license_plate in license_plates:
        x_center, y_center, _, _, _ = license_plate.obb.xywhr.numpy()[0]
        points = license_plate.obb.xyxyxyxy.numpy()[0]
        points = sorted(points, key=lambda x: (x[0], x[1]))
        p1 = points[0] if points[0][1] < points[1][1] else points[1]
        p2 = points[2] if points[2][1] < points[3][1] else points[3]
        a, b = abs(p2[0] - p1[0]), p2[1] - p1[1]
        alpha = math.atan(b / a) * 180 / math.pi
        img = image.copy()
        imgrotate, points = rotate_image(img, (x_center, y_center), alpha, points)
        x_min    = int(min(points[0][0], points[1][0]))
        x_min    = 0 if x_min < 0 else x_min
        y_min     = int(min(points[0][1], points[1][1]))
        y_min     = 0 if y_min < 0 else y_min
        x_max   = int(max(points[2][0], points[3][0]))
        x_max   = 0 if x_max < 0 else x_max
        y_max  = int(max(points[2][1], points[3][1]))
        y_max  = 0 if y_max < 0 else y_max
        license_plate_crop = imgrotate[y_min:y_max, x_min:x_max,:] 
        krs = min(license_plate_crop.shape[0]/640, license_plate_crop.shape[1]/640) if (license_plate_crop.shape[1] / license_plate_crop.shape[0] < 2.0) else min(license_plate_crop.shape[0]/160, license_plate_crop.shape[1]/160)
        license_plate_crop = cv2.resize(license_plate_crop, (int(license_plate_crop.shape[1]/krs), int(license_plate_crop.shape[0]/krs)))
        return license_plate_crop

# Hàm nhận diện ký tự từ biển số
def recognize_plate(image):
    matrix = []
    predicted_characters = ""
    results = number_detector(image)[0]
    for result in results.boxes.data.tolist():
        x,y,w,h,conf,cls = result
        matrix.append([x,y,w,h,cls])
    # Sort the matrix by the x-coordinate of the bounding boxes
    matrix = sorted(matrix, key=lambda x: (x[1] > image.shape[0] // 3, x[0]))
    for i, e in enumerate(matrix):
        x, y, w, h, id = e
        x1, y1 = (x + w) / 2, (y + h) / 2
        if i > 0 and x1 > matrix[i - 1][0] and x1 < matrix[i - 1][2]:
            matrix.pop(i)
            continue
        crop_img = image[int(y):int(h), int(x):int(w)]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        ret, thresh = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.resize(thresh, (28, 28))
        thresh = cv2.medianBlur(thresh, 3)
        thresh = thresh.reshape((28, 28, 1))
        #lưu vào data/result 
        # cv2.imwrite("./data/result/{}.jpg".format(i), thresh)
        img_thresh = img_to_array(thresh)
        img_thresh = np.expand_dims(img_thresh, axis=0)
        img_thresh /= 255.0
        prediction = model.predict(img_thresh)
        predicted_label = np.argmax(prediction)
        predicted_char = text_char[predicted_label]
        predicted_characters += predicted_char
    return predicted_characters

# hàm chính
def main():
    st.title("Nhận diện biển số")
    # upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image_plate = detect_plate(image)
        image = detect_plate_normal(image)
        result = recognize_plate(image_plate)
        cv2.imwrite("static/image.jpg", image)
        cv2.imwrite("static/image_plate.jpg", image_plate)
        st.markdown("### Biển số đã nhận phát hiện")
        st.image(image, caption="Biển số đã nhận diện", use_column_width=True)
        st.markdown("### Biển số đã nhận diện")
        st.image(image_plate, caption="Biển số đã nhận diện", use_column_width=True)
        st.markdown("### Kết quả nhận diện")
        st.write(result)

if __name__ == "__main__":
    main()