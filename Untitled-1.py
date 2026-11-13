import cv2
from ultralytics import YOLO

slot_model = YOLO(
    r"C:\Users\asus\Desktop\parking spot project\best.pt"
)
car_model = YOLO("yolov8n.pt")

image_path = (r"C:\Users\asus\Desktop\parking spot project\par4.png")
frame = cv2.imread(image_path)
if frame is None:
    print("Error: Image not found")
    exit()

slot_results = slot_model.predict(
    frame,
    conf=0.25,
    iou=0.45,
    verbose=False
)

car_results = car_model.predict(
    frame,
    classes=[2],
    conf=0.30,
    iou=0.45,
    verbose=False
)
car_boxes = []
for result in car_results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0]
        )
        car_boxes.append((x1, y1, x2, y2))
        
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 165, 0),
            2
        )

occupied_count = 0
empty_count = 0
slot_number = 0
for result in slot_results:
    for box in result.boxes:
        slot_number += 1
        sx1, sy1, sx2, sy2 = map(
            int, box.xyxy[0]
        )
        slot_area = max(
            1,
            (sx2 - sx1) * (sy2 - sy1)
        )
        is_occupied = False
        for cx1, cy1, cx2, cy2 in car_boxes:
           
            ix1 = max(sx1, cx1)
            iy1 = max(sy1, cy1)
            ix2 = min(sx2, cx2)
            iy2 = min(sy2, cy2)
            overlap_ratio = 0
            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (
                    (ix2 - ix1) * (iy2 - iy1)
                )
                overlap_ratio = (
                    intersection_area / slot_area
                )
            
            car_center_x = (cx1 + cx2) // 2
            car_bottom_y = cy2
            center_inside = (
                sx1 <= car_center_x <= sx2
                and sy1 <= car_bottom_y <= sy2
            )
            
            if center_inside or overlap_ratio >= 0.25:
                is_occupied = True
                break
        
        if is_occupied:
            color = (0, 0, 255)
            status = "Occupied"
            occupied_count += 1
        else:
            color = (0, 255, 0)
            status = "Empty"
            empty_count += 1
        
        cv2.rectangle(
            frame,
            (sx1, sy1),
            (sx2, sy2),
            color,
            2
        )
        
        label = f"Slot {slot_number}: {status}"
        cv2.putText(
            frame,
            label,
            (sx1, max(20, sy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2
        )

total_slots = occupied_count + empty_count
cv2.rectangle(
    frame,
    (10, 10),
    (300, 125),
    (30, 30, 30),
    -1
)
cv2.putText(
    frame,
    f"Total Slots: {total_slots}",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
)
cv2.putText(
    frame,
    f"Occupied: {occupied_count}",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 255),
    2
)
cv2.putText(
    frame,
    f"Empty: {empty_count}",
    (20, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),2
)

height, width = frame.shape[:2]
scale = min(960 / width, 540 / height)
display_frame = cv2.resize(
    frame,
    (int(width * scale), int(height * scale))
)
print("SMART PARKING SYSTEM")
print("--------------------")
print(f"Total Slots: {total_slots}")
print(f"Occupied: {occupied_count}")
print(f"Empty: {empty_count}")
print(f"Detected Cars: {len(car_boxes)}")
cv2.namedWindow(
    "Smart Parking System",
    cv2.WINDOW_NORMAL
)
cv2.imshow(
    "Smart Parking System",
    display_frame
)
cv2.waitKey(0)
cv2.destroyAllWindows()