from ultralytics import YOLO
import cv2
import easyocr
from datetime import datetime

model = YOLO(r"C:\Users\asus\Desktop\parking spot project\my_model.pt")

image_path = (r"C:\Users\asus\Desktop\parking spot project\car3.png")
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}. Please check the file name and extension.")
else:
    reader = easyocr.Reader(['en'])
    results = model(frame)
    annotated_image = results[0].plot()

    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            
            ocr_results = reader.readtext(plate_crop)
            
            for (bbox, text, prob) in ocr_results:
                print(f"Extracted Text: {text} (Confidence: {prob:.2f})")
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(r"C:\Users\asus\Desktop\parking spot project\car_plate_data.txt", "a") as f:
                    f.write(f"Detected Text: {text}, Timestamp: {current_time}\n")
                
                cv2.putText(annotated_image, text, (x1, y1 - 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 7)

    resized_image = cv2.resize(annotated_image, (960, 540))
    cv2.imshow("Saudi Plates Detection", resized_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()