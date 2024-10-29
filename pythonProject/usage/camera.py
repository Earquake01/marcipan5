from emotion_video_track import EmotionVideoTrack
import cv2
import os

if __name__ == "__main__":
    output_folder = "output_preds"
    output_video_emotion_annot = "face_detections_camera.csv"
    output_video_emotion_viz = "emotion_prediction_viz_camera.mp4"

    emotion_video_track = EmotionVideoTrack()

    # Инициализация камеры
    cap = cv2.VideoCapture(0)  # 0 означает использование основной камеры

    # Получение параметров видео с камеры
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Создание временного файла для хранения видео с камеры
    temp_video_path = "test_video/temp_camera_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    print("Запись видео с камеры. Нажмите 'q' для остановки.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов камеры
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Обработка записанного видео...")
    # Обработка записанного видео с помощью EmotionVideoTrack
    emotion_video_track.track_emotions_on_video(
        temp_video_path, output_folder, output_video_emotion_annot, output_video_emotion_viz
    )

    # Удаление временного файла
    os.remove(temp_video_path)

    print("Обработка завершена. Результаты сохранены в указанных файлах.")