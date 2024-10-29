import os

import gdown
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

import emotion_net_infer.transforms as transforms


class EmotionSeqClassifier:
    """Класс для инференса нейронной сети для аудиовизуального определения
    эмоций человека.
    Основан на multimodal-emotion-recognition
    (https://github.com/katerynaCh/multimodal-emotion-recognition)"""

    def __init__(self):
        self.video_transform = transforms.Compose([transforms.ToTensor(255)])
        self.frame_sequence = 15
        self.class_labels = [
            "neutral",
            "calm",
            "happy",
            "sad",
            "angry",
            "fearful",
            "disgust",
            "surprised",
        ]
        self.input_clip_shape = [15, 3, 224, 224]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # скачиваем готовую модель из Google Drive
        model_path = "./model/intermediate_attention.pth"

        if os.path.exists(model_path):
            self.model = torch.load(model_path, map_location=self.device)
        else:
            url = "https://drive.google.com/drive/folders/1Mvt6kzjKQDraOhP-LABLF_EAyowXqRSA"
            self.output_path = gdown.download_folder(url)[0]

        # скачиваем классы модели без которых модель не загружается
        models_path = "./models"

        if os.path.exists(models_path):
            pass
        else:
            url = "https://drive.google.com/drive/folders/1S-GtpnvXH9Vm219qjgtI2kysEV6h1qMr"
            gdown.download_folder(url)[0]

        # модель
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

        print(self.device)
        if self.device == "cpu":
            self.model = self.model.module.to(self.device)
        else:
            self.model.to(self.device)

    @staticmethod
    def load_image(image_path):
        """Загружаем картинку

        Parameters
        ----------
        image_path : str
            путь к картинке и датасета

        Returns
        -------
        np.array
            2d np.array представляющий собой картинку считаную PIL
        """
        img_raw = Image.open(image_path)
        img_raw.load()

        return img_raw

    def prepare_pil_imgs(self, pil_images):
        imgs = [np.array(pil_image) for pil_image in pil_images]

        # переводим rgb в bgr
        imgs = [img[:, :, ::-1] for img in imgs]

        # уменьшаем до размера нужного модели
        imgs = [
            Image.fromarray(img, "RGB").resize(
                self.input_clip_shape[2:], resample=Image.BILINEAR
            )
            for img in imgs
        ]

        # модель предсказывает на последовательности из
        # self.frame_sequence кадров вырезанных лиц.
        # сгенерируем последовательность из self.frame_sequence
        # одинаковых PIL image
        if len(imgs) == 1:
            img = imgs[0]
            for i in range(1, 15):
                imgs.append(img)

        if len(imgs) != self.frame_sequence:
            raise ValueError(
                f"Для предсказания нужно отдать в модель"
                f"{self.frame_sequence} кадров сразу или только 1 кадр"
            )

        # преобразуем в Torch.Tensor
        if self.video_transform is not None:
            self.video_transform.randomize_parameters()
            clip = [self.video_transform(img) for img in imgs]

        inputs_visual = torch.stack(clip, 0)

        return inputs_visual

    def predict(self, inputs_visual, inputs_audio, modality="video", dist="zeros"):
        assert modality in ["both", "audio", "video"]

        if modality == "audio":
            print("Skipping video modality")
            if dist == "noise":
                print("Evaluating with full noise")
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == "addnoise":
                print("Evaluating with noise")
                inputs_visual = inputs_visual + (
                    torch.mean(inputs_visual)
                    + torch.std(inputs_visual) * torch.randn(inputs_visual.size())
                )
            elif dist == "zeros":
                inputs_visual = torch.zeros(inputs_visual.size())
            else:
                print("UNKNOWN DIST!")

        elif modality == "video":
            if dist == "noise":
                print("Evaluating with noise")
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == "addnoise":
                print("Evaluating with added noise")
                inputs_audio = inputs_audio + (
                    torch.mean(inputs_audio)
                    + torch.std(inputs_audio) * torch.randn(inputs_audio.size())
                )

            elif dist == "zeros":
                inputs_audio = torch.zeros(inputs_audio.size())

        with torch.no_grad():
            inputs_visual = Variable(inputs_visual.to(self.device))
            inputs_audio = Variable(inputs_audio.to(self.device))

        outputs = self.model(inputs_audio, inputs_visual)
        scores = torch.softmax(outputs, dim=1)
        emotion_class = torch.argmax(scores[0])
        score = scores[0][emotion_class].detach().to("cpu").item()
        emotion_class = emotion_class.detach().to("cpu").item()

        return emotion_class, score

    def predict_visual_only(self, inputs_visual):
        # имитируем тензор аудио признаков из нулей нужной размерности
        inputs_audio = torch.zeros([1, 10, 156])

        emotion_class, score = self.predict(inputs_visual, inputs_audio)

        return emotion_class, score

    def predict_on_images(self, pil_images, return_label=False):
        """Предсказать моделью эмоцию человека на последовательности
        из self.frame_sequence кадров (или 1 кадре) с вырезанным лицом
        от FaceDetector
        (см. https://gitlab.com/group_19200719/pytorch_retinaface_infer)

        Parameters
        ----------
        pil_images : list of PIL.Image
            Список с PIL.Image. Каждый PIL.Image содержит вырезанное
            из изображение лицо человека от FaceDetector.
            Список либо содержит 15 кадров лица одного человека следующих
            друг за другом
            Либо содержит ровно 1 кадр с вырезанным лицом.
            В этом случае будет составлена последовательность из 15
            одинаковых кадров.
            Это нужно, чтобы модель могла предсказывать на 1 картинке.
            Но в целом, она предназначена для работы с последовательностями
            кадров и это ее основной режим работы.
        return_label : bool, optional
            Если True то возвращаем текстовое описание классов вместо их номеров,
            by default False

        Returns
        -------
        int(by default) or str(if return_label==True)
            Возвращает номер класса распознанной эмоции, by default
            Если return_label==True, - вернет текстовое описание класса
        """
        inputs_visual = self.prepare_pil_imgs(pil_images)
        emotion_class, score = self.predict_visual_only(inputs_visual)

        if return_label:
            emotion_class = self.class_labels[emotion_class]

        return emotion_class, score

    def predict_on_image(self, pil_image, return_label=False):
        """Предсказать моделью эмоцию человека на последовательности
        длинной  1 кадре) с вырезанным лицом
        от FaceDetector
        (см. https://gitlab.com/group_19200719/pytorch_retinaface_infer)

        Parameters
        ----------
        pil_images : list of PIL.Image
            Список с PIL.Image. Каждый PIL.Image содержит вырезанное
            из изображение лицо человека от FaceDetector.
            Список либо содержит 15 кадров лица одного человека следующих
            друг за другом
            Либо содержит ровно 1 кадр с вырезанным лицом.
            В этом случае будет составлена последовательность из 15
            одинаковых кадров.
            Это нужно, чтобы модель могла предсказывать на 1 картинке.
            Но в целом, она предназначена для работы с последовательностями
            кадров и это ее основной режим работы.
        return_label : bool, optional
            Если True то возвращаем текстовое описание классов вместо их номеров,
            by default False

        Returns
        -------
        int(by default) or str(if return_label==True)
            Возвращает номер класса распознанной эмоции, by default
            Если return_label==True, - вернет текстовое описание класса
        """
        return self.predict_on_images([pil_image], return_label)
