import cv2
import torch
import pymysql
import serial
import pynmea2
import time
from haversine import haversine
from datetime import datetime

serial_port = '/dev/ttyACM0'  # 실제 장치에 맞게 포트 확인
baud_rate = 115200  # ZED-F9P의 기본 보드레이트

ser = serial.Serial(serial_port, baud_rate, timeout=1)

# yolov5 설정
custom_path = './best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_path, device=device)

save_delay = 5
last_save_time = 0
cap = cv2.VideoCapture(2)

import pymysql

# 데이터베이스 연결
db = pymysql.connect(
	host='34.172.241.206',
	user='test',
	password='123456',
	database='capstone'
)
cursor = db.cursor()

# get GPS value
def get_gps_data(image, time):
	while True:
		try:
			line = ser.readline().decode('utf-8', errors='replace').strip()

			if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
				try:
					msg = pynmea2.parse(line)

					if msg.gps_qual == 0:
						print("No GPS fix, unable to obtain coordinates.")
					else:
						latitude = msg.latitude
						longitude = msg.longitude

						print(f"Latitude: {latitude}, Longitude: {longitude}")
						save_data(image, time, latitude, longitude)
						return latitude, longitude, msg.altitude
				except pynmea2.ParseError as e:
					print(f"Error parsing NMEA data: {e}")
		except pynmea2.ParseError as e:
			print(f"Error parsing NMEA data: {e}")
		except Exception as e:
			print(f"Error: {e}")

# 데이터 저장
def save_data(image, time, x, y):
	# 근처 좌표 값이 이미 저장 되어 있을 경우 저장 안 함
	if is_duplicate(x, y):
		print("duplicate")
	else:
		_, img_encoded = cv2.imencode('.jpg', image)
		img_bytes = img_encoded.tobytes()

		query = "insert into banner(image, banner_time, latitude, longitude) values (%s, %s, %s, %s);"
		cursor.execute(query, (img_bytes, time, x, y))
		db.commit()
		print("success!")

# 중복 감지를 위한 기존 좌표 조회
def is_duplicate(x, y):
	query = "select latitude, longitude from banner;"
	cursor.execute(query)

	for (ex, ey) in cursor.fetchall():
		if is_similar(x, y, float(ex), float(ey)):
			return True
	return False

# 좌표 비교, 좌표 간 거리가 최소 5m 이상 되도록 함
def is_similar(x1, y1, x2, y2, threshold=5.0):
	one = (x1, y1)
	two = (x2, y2)
	distance = haversine(one, two, unit='m')
	print(distance)
	return distance < threshold

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		print("Failed to grab frame")
		break

	frame = cv2.resize(frame, (640, 480))

	# 객체 탐지
	results = model(frame)
	labels, confidences, boxes = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, -2].cpu().numpy(), results.xyxyn[0][:, :-2].cpu().numpy()

	# 현수막과 현수막 거치대의 유무를 확인
	detected_classes = [model.names[int(x)] for x in labels]

	for i, label in enumerate(labels):
		class_name = model.names[int(label)]
		confidence = confidences[i]
		box = boxes[i]
		x1, y1, x2, y2 = box
		x1 = int(x1*frame.shape[1])
		y1 = int(y1*frame.shape[0])
		x2 = int(x2*frame.shape[1])
		y2 = int(y2*frame.shape[0])

		# 카메라 화면에 객체 경계 상자 표시
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	if 'banner' in detected_classes and 'bnner_holder' in detected_classes:
		print("banner and holder detected")
	elif 'banner' in detected_classes and 'banner_holder' not in detected_classes:
		current_time = time.time()

		# 최소 5초마다 데이터가 저장되도록 설정
		if current_time - last_save_time > save_delay:
			print("Only banner detected")
			timestamp = datetime.now()
			get_gps_data(frame, timestamp)
			last_save_time = current_time

	cv2.imshow('YOLOv5', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
