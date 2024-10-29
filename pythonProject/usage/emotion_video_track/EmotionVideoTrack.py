import os
import numpy as np
import pandas as pd
from emotion_det_net_infer import FaceDetector
from emotion_net_infer import EmotionSeqClassifier
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

from .sort_source import Sort


class EmotionVideoTrack:
    """Класс отслеживающий лица людей на видео с помощью SORT-трекинга,
    и определяющий эмоции каждого из них для каждых 15 кадров видео.
    Для определения эмоций используется нейронная сеть для
    аудиовизуального определения эмоций человека.
    Основан на multimodal-emotion-recognition
    (https://github.com/katerynaCh/multimodal-emotion-recognition)"""

    def __init__(self):
        self.facedet = FaceDetector()
        self.emo_seq_classifier = EmotionSeqClassifier()

    def track_emotions_on_video(
        self,
        video_path,
        output_folder,
        output_video_emotion_annot,
        output_video_emotion_viz,
        st_progress=None
    ):
        """Метод
        * обнаруживает и отслеживает лицо каждого человека на видео индивидуально
        * распознает эмоции для каждого человека на последовательности из 15 кадров
        * сохранит таблицу с результатами распознавания в csv-таблицу
        * сохранит видео с визуализацией распознанных эмоций
        

        Parameters
        ----------
        video_path : str
            путь к видео с записью онлайн конференции с лицами людей (Zoom, Skype, и т.д.)
        output_folder : str
            путь к папке для сохранения результатов работы
        output_video_emotion_annot : str
            название файла с таблицей с результатами распознавания
        output_video_emotion_viz : str
            название видео с визуализацией распознанных эмоций
        """        
        # создаем папку для сохранения лиц людей, эмоции которых распознаются
        persons_faces_dir = os.path.join(output_folder, "persons_faces")
        if not os.path.exists(persons_faces_dir):
            os.makedirs(persons_faces_dir)

        clip = VideoFileClip(video_path)
        fps = clip.fps
        total_frames = int(fps * clip.duration)

        mot_tracker = Sort()

        # обнаруживаем лица людей на видео
        persons_faces = {}
        for num, frame in enumerate(clip.iter_frames()):
            image = Image.fromarray(frame.astype("uint8"), "RGB")
            print(f"Детекция лиц на {num + 1} из {total_frames} кадров видео")

            face_detections = self.facedet.detect_faces(image)
            trackers = mot_tracker.update(face_detections)

            for track in trackers:
                person_id = f"person_{int(track[-1])}"
                if person_id not in persons_faces:
                    persons_faces[person_id] = [np.nan] * num

                    face_crop = image.crop(track[:-1])
                    face_crop_path = os.path.join(persons_faces_dir, f"{person_id}.jpg")
                    face_crop.save(face_crop_path)

                persons_faces[person_id].append(track[:-1])
                
            if st_progress is not None:
                st_progress.progress((num / total_frames)*0.9, text='Детектируем лица')

            if num == total_frames - 1:
                break

        for person_id in persons_faces:
            if len(persons_faces[person_id]) < total_frames:
                num_of_nuns = total_frames - len(persons_faces[person_id])
                persons_faces[person_id].extend([np.nan] * num_of_nuns)

        persons_ids = list(persons_faces.keys())
        for person_id in persons_ids:
            person_id_emo = f"{person_id}_emo"
            person_id_score = f"{person_id}_score"
            persons_faces[person_id_emo] = []
            persons_faces[person_id_score] = []

        persons_crop_faces = {}
        for person_id in persons_faces:
            persons_crop_faces[person_id] = []

        # распознаем эмоции людей
        for num, frame in enumerate(clip.iter_frames()):
            image = Image.fromarray(frame.astype("uint8"), "RGB")
            for person_id in persons_ids:
                face_detections = persons_faces[person_id][num]
                person_id_emo = f"{person_id}_emo"
                person_id_score = f"{person_id}_score"

                if isinstance(face_detections, np.ndarray):
                    face_crop = image.crop(face_detections)
                    persons_crop_faces[person_id].append(face_crop)

                    if len(persons_crop_faces[person_id]) == 15:
                        face_crops = persons_crop_faces[person_id]
                        persons_crop_faces[person_id] = []
                        emo_cls, score = self.emo_seq_classifier.predict_on_images(
                            face_crops, return_label=True
                        )

                        persons_faces[person_id_emo].extend([emo_cls] * 15)
                        persons_faces[person_id_score].extend([score] * 15)
                else:
                    persons_faces[person_id_emo].append(np.nan)
                    persons_faces[person_id_score].append(np.nan)
                    persons_crop_faces[person_id] = []

            if st_progress is not None:
                st_progress.progress(93, text='Распознали эмоции')
            
            if num == total_frames - 1:
                break
        
        # записываем результаты распознавания эмоций в csv-файл
        max_len = 0
        for person_id in persons_faces:
            if len(persons_faces[person_id]) > max_len:
                max_len = len(persons_faces[person_id])

        for person_id in persons_faces:
            if len(persons_faces[person_id]) < max_len:
                num_of_nuns = max_len - len(persons_faces[person_id])
                persons_faces[person_id].extend([np.nan] * num_of_nuns)

        faces_df = pd.DataFrame.from_dict(persons_faces)
        faces_df = faces_df.reindex(sorted(faces_df.columns), axis=1)

        output_csv_path = os.path.join(output_folder, output_video_emotion_annot)
        faces_df.to_csv(output_csv_path, index=False)
        
        if st_progress is not None:
            st_progress.progress(96, text='Получили файл разметки для видео')

        # сохраняем видео с визуализацией распознанных эмоций
        clip_frames = []
        for num, frame in enumerate(clip.iter_frames()):
            image = Image.fromarray(frame.astype("uint8"), "RGB")

            for person_id in persons_ids:
                person_id_emo = f"{person_id}_emo"
                person_id_score = f"{person_id}_score"

                emotion_cls = persons_faces[person_id_emo][num]
                score = persons_faces[person_id_score][num]
                detection = persons_faces[person_id][num]

                if not np.isnan(score):
                    image = self.facedet.viz_detections_by_data(
                        image, detection, emotion_cls, score
                    )

            clip_frames.append(np.array(image))

            if num == total_frames - 1:
                break

        clip = ImageSequenceClip(clip_frames, fps)
        output_video_path = os.path.join(output_folder, output_video_emotion_viz)
        clip.write_videofile(output_video_path)
        
        if st_progress is not None:
            st_progress.progress(100, text='Получили видео с визуализацией эмоций')
