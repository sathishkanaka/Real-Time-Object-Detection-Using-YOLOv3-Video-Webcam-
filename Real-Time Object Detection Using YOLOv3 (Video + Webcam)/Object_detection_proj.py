import cv2
import numpy as np

# Load YOLO model and classes
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Object Detection Function
def detect_objects(frame, net, output_layers, classes):
    height, width, _ = frame.shape

    # Create a blob and perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                 (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

    # Extract bounding boxes and confidences
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

# Processing Function (Video or Webcam)
def process_video(source=0):
    net, classes, output_layers = load_yolo()

    # Define vehicle-related COCO labels
    vehicle_labels = ["car", "motorbike", "bus", "truck", "bicycle"]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot read frame.")
            break

        boxes, confidences, class_ids, indexes = detect_objects(frame, net, output_layers, classes)

        # Counters
        person_count = 0
        vehicle_count = 0
        other_count = 0
        total_objects = 0

        # Draw detections and count
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                total_objects += 1

                # Categorize detections
                if label.lower() == "person":
                    person_count += 1
                    color = (0, 0, 255)  # Red
                elif label.lower() in vehicle_labels:
                    vehicle_count += 1
                    color = (255, 0, 0)  # Blue
                else:
                    other_count += 1
                    color = (0, 255, 0)  # Green

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display counts on screen
        cv2.putText(frame, f"Persons: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Others: {other_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Objects: {total_objects}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("YOLOv3 Object Detection (Webcam/Video)", frame)

        # Press ESC to exit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
# Program Entry Point

if __name__ == "__main__":
    print("=== YOLOv3 Object Detection System ===")
    print("1. Live Webcam")
    print("2. Upload Recorded Video")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("Starting webcam...")
        process_video(0)
    elif choice == "2":
        path = input("Enter full path of video file (e.g., D:/Videos/sample.mp4): ").strip()
        process_video(path)
    else:
        print("Invalid choice. Please run again.")
