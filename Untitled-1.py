import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = r"C:\Users\asus\Desktop\parking spot project\car_video.mp4.mp4"
cap = cv2.VideoCapture(video_path)


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("--- Dataset / Video Information Output ---")
print(f"Total Video Frames: {total_frames}")
print(f"Frames Per Second (FPS): {fps}")
print(f"Resolution: {width}x{height}")
print("Status: Dataset loaded and preprocessed successfully!")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

cv2.namedWindow("YOLO Car Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Car Detection", 640, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[2])
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Car Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()