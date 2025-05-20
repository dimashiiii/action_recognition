import cv2
from ultralytics import YOLO

# Загружаем модель YOLO-Pose
model = YOLO("yolov8n-pose.pt")

# Захватываем видео с веб-камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Не удалось открыть камеру.")
    exit()

print("✅ Камера запущена. Нажми 'q' чтобы выйти.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Получаем результаты предсказания
    results = model(frame)

    # Получаем ключевые точки из первого объекта
    for result in results:
        keypoints = result.keypoints
        if keypoints is not None:
            for kp in keypoints.xy:  # xy: (n_persons, 17, 2)
                for x, y in kp:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Показываем кадр
    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершаем
cap.release()
cv2.destroyAllWindows()
