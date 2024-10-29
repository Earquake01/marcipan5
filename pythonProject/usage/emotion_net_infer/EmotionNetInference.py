import os

import gdown
import mlflow
import torch
from PIL import Image
from torchvision import transforms


class EmotionNet:
    """Класс для инференса нейронной сети для предсказания эмоция человека."""

    def __init__(self, mlflow_server_uri="", model_stage="Production", model_path=""):
        """Инициализатор класса.

        Parameters
        ----------
        mlflow_server_uri : str
            адрес сервера и порт с MLFlow, например http://100.100.100.100:9000
        model_stage : str
            название стадии разработки модели, которую мы хотим забрать из MLFlow.
            По умолчанию, берется последняя модель версии "Production".
        """
        model_name = "emotion_classifier"

        # взять модель из MLFlow (который под VPN)
        if mlflow_server_uri:
            mlflow.set_tracking_uri(mlflow_server_uri)
            self.model = mlflow.pytorch.load_model(
                f"models:/{model_name}/{model_stage}"
            )

        # взять модель из файла модели если он указан
        elif model_path:
            self.model = torch.load(model_path)

        # если MlFLow Registry недоступен (для любого внешнего разработчика)
        # скачиваем последнюю модель из Google Drive
        else:
            model_path = os.path.join(
                os.getcwd(), "emotion_cls_prod_model/last_prod_model.pth"
            )
            if os.path.exists(model_path):
                self.model = torch.load(model_path)
            else:
                url = "https://drive.google.com/drive/folders/1A2k4dE0Av2IcDHUurXOHN3Dy8OA9xlUQ"
                self.output_path = gdown.download_folder(url)[0]
                self.model = torch.load(self.output_path)

        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # для преобразования изображений в тензоры PyTorch и нормализации входа
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.model.dim[1:]),
                transforms.ToTensor(),
                # mean и std для набора данных ImageNet на котором были обучены
                # предобученные сети из torchvision
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @staticmethod
    def load_image(path2image):
        """Загружаем картинку

        Parameters
        ----------
        path2image : str
            путь к картинке и датасета

        Returns
        -------
        np.array
            2d np.array представляющий собой картинку считаную Pil
        """
        image = Image.open(path2image)
        image.load()
        return image

    def preprocess_image(self, pil_image):
        """Выполняет предобработку изображения PIL для нейронной сети.

        Parameters
        ----------
        pil_image : np.array
            2d np.array представляющий собой картинку считаную PIL

        Returns
        -------
        torch.tensor
            batch из одной единственной картинки на которой
            мы хоти сделать предсказание.
        """
        image = self.transform(pil_image)
        # создаем батч из одной картинки
        image = image[None, :, :, :]
        return image

    def predict_on_image(self, image, return_label=False):
        """Предсказываем на картинке считаной PIL.

        Parameters
        ----------
        image : np.array
            2d np.array представляющий собой картинку считаную PIL
        return_label : bool, optional
            флаг на возвращение названия класса в виде строки
            вместо номера класса, by default False

        Returns
        -------
        (emotion_class, score)
            emotion_class : int (or str if return_label==True)
                номера класса или если return_label==True,
                названия класса в виде строки
            score : float
                вероятность принадлежности именно этому классу
        """
        image = self.preprocess_image(image).to(self.device)

        with torch.no_grad():
            pred = self.model(image)
            emotion_class = torch.argmax(pred, dim=1).tolist()[0]
            score = torch.softmax(pred, dim=1).flatten().tolist()[emotion_class]

            if return_label:
                emotion_class = self.model.class_labels[emotion_class]

        return emotion_class, score

    def load_image_and_predict(self, path2image, return_label=False):
        """Предсказываем на картинке расположенной по пути path2image.

        Parameters
        ----------
        path2image : str
            путь к картинке на которой мы хотим осуществить предсказание
        return_label : bool, optional
            флаг на возвращение названия класса в виде строки
            вместо номера класса, by default False

        Returns
        -------
        (emotion_class, score)
            emotion_class : int (or str if return_label==True)
                номера класса или если return_label==True,
                названия класса в виде строки
            score : float
                вероятность принадлежности именно этому классу
        """
        image = self.load_image(path2image)
        emotion_class, score = self.predict_on_image(image, return_label)

        return emotion_class, score
