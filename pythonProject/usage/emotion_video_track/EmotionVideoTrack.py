import os
import numpy as np
import pandas as pd
from emotion_det_net_infer import FaceDetector
from emotion_net_infer import EmotionSeqClassifier
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image


class EmotionVideoTrack:
    """Класс отслеживающий лица людей на видео с помощью SORT-трекинга,
    и определяющий эмоции каждого из них для каждых 15 кадров видео.
    Для определения эмоций используется нейронная сеть для
    аудиовизуального определения эмоций человека.
    """

    def __init__(self):
        self.facedet = FaceDetector()
        self.emo_seq_classifier = EmotionSeqClassifier()

    def track_emotions_on_video(
            self,
            video_path,
            output_folder,
            annotations_file,
            output_video_emotion_viz,
            progress_callback=None
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
        annotations_file : str
            название файла с таблицей с результатами распознавания
        output_video_emotion_viz : str
            название видео с визуализацией распознанных эмоций
        progress_callback : function, optional
            Callback function to update progress.  Takes frame_num and total_frames.
        """

        persons_faces_dir = os.path.join(output_folder, "persons_faces")
        os.makedirs(persons_faces_dir, exist_ok=True)  # Создаем папку, если ее нет

        clip = VideoFileClip(video_path)
        fps = clip.fps
        total_frames = int(fps * clip.duration)
        print(f"Общее количество кадров: {total_frames}")

        # Инициализируем трекер (предположим, что Sort уже импортирован)
        from .sort_source import Sort
        mot_tracker = Sort()

        # Загружаем существующие данные (если они есть)
        existing_faces = self._load_existing_faces(persons_faces_dir)
        persons_faces = existing_faces.copy() if existing_faces is not None else {}
        existing_ids = list(persons_faces.keys())

        # Устанавливаем начальный ID
        next_id = self._get_next_id(existing_ids)

        # Обнаруживаем лица людей на видео
        for num, frame in enumerate(clip.iter_frames()):
            image = Image.fromarray(frame.astype("uint8"), "RGB")
            print(f"Детекция лиц на {num + 1} из {total_frames} кадров видео")

            if progress_callback:
                progress_callback(num, total_frames)

            face_detections = self.facedet.detect_faces(image)
            trackers = mot_tracker.update(face_detections)

            for track in trackers:
                person_id_int = int(track[-1])
                person_id = f"person_{person_id_int}"

                if person_id not in persons_faces:
                    persons_faces[person_id] = [np.nan] * num
                    #сохраняет лицо человека
                    face_crop = image.crop(track[:-1])
                    face_crop_path = os.path.join(persons_faces_dir, f"{person_id}.jpg")
                    face_crop.save(face_crop_path)

                persons_faces[person_id].append(track[:-1])

            if num == total_frames - 1:
                break

        # Заполняем NaN в конце каждого трека
        for person_id in persons_faces:
            if len(persons_faces[person_id]) < total_frames:
                num_of_nans = total_frames - len(persons_faces[person_id])
                persons_faces[person_id].extend([np.nan] * num_of_nans)

        # Если есть новые лица, добавляем их в существующие ID
        persons_ids = list(persons_faces.keys())

        for person_id in persons_ids:
            person_id_emo = f"{person_id}_emo"
            person_id_score = f"{person_id}_score"
            if person_id_emo not in persons_faces:
                persons_faces[person_id_emo] = [np.nan] * total_frames
                persons_faces[person_id_score] = [np.nan] * total_frames

        persons_crop_faces = {}
        for person_id in persons_faces:
            persons_crop_faces[person_id] = []

        # Распознаем эмоции людей
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
                        persons_faces[person_id_emo][num - 14:num + 1] = [emo_cls] * 15  # Записываем эмоции
                        persons_faces[person_id_score][num - 14:num + 1] = [score] * 15  # Записываем score
                else:
                    pass

            if num == total_frames - 1:
                break

        # Записываем результаты распознавания эмоций в csv-файл
        faces_df = pd.DataFrame.from_dict(persons_faces)
        faces_df = faces_df.reindex(sorted(faces_df.columns), axis=1)

        output_csv_path = os.path.join(output_folder, annotations_file)
        self._save_face_data(faces_df, output_csv_path)

        # Сохраняем видео с визуализацией распознанных эмоций
        clip_frames = []
        for num, frame in enumerate(clip.iter_frames()):
            image = Image.fromarray(frame.astype("uint8"), "RGB")

            for person_id in persons_ids:
                person_id_emo = f"{person_id}_emo"
                person_id_score = f"{person_id}_score"

                if person_id_emo in persons_faces and person_id_score in persons_faces and person_id in persons_faces:
                    emotion_cls = persons_faces[person_id_emo][num]
                    score = persons_faces[person_id_score][num]
                    detection = persons_faces[person_id][num]

                    if not np.isnan(self._safe_float(score)):
                        if isinstance(detection, np.ndarray):
                            image = self.facedet.viz_detections_by_data(
                                image, detection, emotion_cls, score
                            )

            clip_frames.append(np.array(image))

            if num == total_frames - 1:
                break

        clip = ImageSequenceClip(clip_frames, fps)
        output_video_path = os.path.join(output_folder, output_video_emotion_viz)
        clip.write_videofile(output_video_path)

    def _safe_float(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan

    def _load_existing_faces(self, persons_faces_dir):
        """Загружает существующие лица из CSV файла, если он существует."""
        csv_path = os.path.join(persons_faces_dir, "face_detections.csv")
        if os.path.exists(csv_path):
            try:
                faces_df = pd.read_csv(csv_path)
                return faces_df.to_dict(orient='list')
            except pd.errors.EmptyDataError:
                return {}
        else:
            return {}

    def _save_face_data(self, faces_df, output_csv_path):
        """Сохраняет данные о лицах в CSV файл, дополняя его, если он существует."""
        if os.path.exists(output_csv_path):
            try:
                existing_df = pd.read_csv(output_csv_path)
                faces_df = pd.concat([existing_df, faces_df], ignore_index=True)
            except pd.errors.EmptyDataError:  # Handle empty file
                pass  # If the file is empty, just use the current faces_df

        faces_df.to_csv(output_csv_path, index=False)

    def _get_next_id(self, existing_ids):
        """Определяет следующий доступный ID, гарантируя его уникальность."""
        if not existing_ids:
            return 1

        existing_numbers = [int(id.split('_')[-1]) for id in existing_ids]
        next_id = max(existing_numbers) + 1

        return next_id