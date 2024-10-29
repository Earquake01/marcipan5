from emotion_video_track import EmotionVideoTrack

if __name__ == "__main__":
    # путь к видео на котором будем распознавать эмоции
    video_path = "test_video/RAVDESS_zoom_demo.mp4"
    
    # путь к папке для сохранения результатов
    output_folder = "output_preds"
    
    # название файла с полученной разметкой видео
    output_video_emotion_annot = "face_detections1.csv"
    
    # видео с визуализацией распознанных эмоций
    output_video_emotion_viz = "emotion_prediction_viz1.mp4"

    # инициализация класса
    emotion_video_track = EmotionVideoTrack()

    # осуществление предсказания
    emotion_video_track.track_emotions_on_video(
        video_path, output_folder, output_video_emotion_annot, output_video_emotion_viz
    )
