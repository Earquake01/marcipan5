from emotion_video_track import EmotionVideoTrack

if __name__ == "__main__":
    video_path = "test_video/RAVDESS_zoom_demo.mp4"
    output_folder = "output_preds"
    output_video_emotion_annot = "face_detections1.csv"
    output_video_emotion_viz = "emotion_prediction_viz1.mp4"

    emotion_video_track = EmotionVideoTrack()

    emotion_video_track.track_emotions_on_video(
        video_path, output_folder, output_video_emotion_annot, output_video_emotion_viz
    )
