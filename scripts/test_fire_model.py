from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path("models") / "firedetect-11s.pt"
IMAGE_PATH = Path("data") / "samples" / "train_1024.jpg"
OUTPUT_DIR = Path("data") / "samples" / "results"


model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=False,
    verbose=False,
)

print("Model loaded successfully.")
print("Classes:", model.names)

for result in results:
    print(f"\nImage: {IMAGE_PATH.name}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        print(
            f"Class: {model.names[class_id]} | "
            f"Confidence: {confidence:.3f} | "
            f"Box: {coordinates}"
        )

    annotated_image = result.plot()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "fire_test_result.jpg"

    import cv2

    cv2.imwrite(str(output_path), annotated_image)

    print(f"Annotated result saved to: {output_path}")
