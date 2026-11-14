import cv2
import easyocr
from ultralytics import YOLO
from datetime import datetime

path = r"C:\Users\asus\Desktop\parking spot project"

slot_model = YOLO(path + r"\best.pt")
car_model = YOLO("yolov8n.pt")
plate_model = YOLO(path + r"\my_model.pt")
reader = easyocr.Reader(['en'])

frame = cv2.imread(r"C:\Users\asus\Desktop\parking spot project\Screenshot 2026-11-14 082716.png")

if frame is None:
    print("Error: Image not found")
    exit()

original_frame = frame.copy()

slot_results = slot_model.predict(frame, conf=0.25, iou=0.45, verbose=False)
car_results = car_model.predict(frame, classes=[2], conf=0.30, iou=0.45, verbose=False)

car_boxes = [
    tuple(map(int, box.xyxy[0]))
    for result in car_results
    for box in result.boxes
]

occupied_count = 0
empty_count = 0
slot_number = 0

for result in slot_results:
    for box in result.boxes:
        sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
        slot_number += 1
        slot_area = max(1, (sx2 - sx1) * (sy2 - sy1))
        is_occupied = False

        for cx1, cy1, cx2, cy2 in car_boxes:
            ix1, iy1 = max(sx1, cx1), max(sy1, cy1)
            ix2, iy2 = min(sx2, cx2), min(sy2, cy2)

            if ix2 > ix1 and iy2 > iy1:
                overlap = ((ix2 - ix1) * (iy2 - iy1)) / slot_area
                center_inside = (
                    sx1 <= (cx1 + cx2) // 2 <= sx2
                    and sy1 <= cy2 <= sy2
                )

                if center_inside or overlap >= 0.25:
                    is_occupied = True
                    break

        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        status = "Occupied" if is_occupied else "Empty"

        occupied_count += int(is_occupied)
        empty_count += int(not is_occupied)

        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.putText(frame, f"Slot {slot_number}: {status}",
                    (sx1, max(20, sy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

plate_file = path + r"\car_plate_data.txt"

for car_index, (cx1, cy1, cx2, cy2) in enumerate(car_boxes):

    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)

    car_crop = original_frame[cy1:cy2, cx1:cx2]

    if car_crop.size == 0:
        continue

    plate_results = plate_model.predict(car_crop, conf=0.25, verbose=False)

    for result in plate_results:
        for box in result.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])

            plate_crop = car_crop[py1:py2, px1:px2]

            if plate_crop.size == 0:
                continue

            texts = [
                text for _, text, prob in reader.readtext(plate_crop)
                if prob >= 0.25
            ]

            plate_text = " ".join(texts).strip()

            x1, y1 = cx1 + px1, cy1 + py1
            x2, y2 = cx1 + px2, cy1 + py2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            if plate_text:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"Car {car_index + 1}: {plate_text}")

                with open(plate_file, "a", encoding="utf-8") as f:
                    f.write(f"Car: {car_index + 1}, Plate: {plate_text}, Time: {current_time}\n")

                cv2.putText(frame, plate_text, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

total_slots = occupied_count + empty_count

cv2.rectangle(frame, (10, 10), (320, 125), (30, 30, 30), -1)

for y, label, value, color in [
    (35, "Total Slots", total_slots, (255, 255, 255)),
    (70, "Occupied", occupied_count, (0, 0, 255)),
    (105, "Empty", empty_count, (0, 255, 0))
]:
    cv2.putText(frame, f"{label}: {value}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

height, width = frame.shape[:2]
scale = min(960 / width, 540 / height)
display_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

print("\nSMART PARKING SYSTEM")
print("--------------------")
print(f"Total Slots: {total_slots}")
print(f"Occupied: {occupied_count}")
print(f"Empty: {empty_count}")
print(f"Detected Cars: {len(car_boxes)}")

cv2.namedWindow("Smart Parking System", cv2.WINDOW_NORMAL)
cv2.imshow("Smart Parking System", display_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
