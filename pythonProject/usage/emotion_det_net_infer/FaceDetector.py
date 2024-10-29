import os
import gdown
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from .data import cfg_re50
from .layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode
from .utils.nms.py_cpu_nms import py_cpu_nms


def find_font_size(text, font, image, target_width_ratio):
    tested_font_size = 100
    tested_font = ImageFont.truetype(font, tested_font_size)
    observed_width, observed_height = get_text_size(text, image, tested_font)
    estimated_font_size = (
        tested_font_size / (observed_width / image.width) * target_width_ratio
    )
    return round(estimated_font_size)


def get_text_size(text, image, font):
    im = Image.new("RGB", (image.width, image.height))
    draw = ImageDraw.Draw(im)
    # Используем textbbox для получения границ текста
    bbox = draw.textbbox((0, 0), text, font=font)
    # Вычисляем ширину и высоту текста
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print("Missing keys:{}".format(len(missing_keys)))
    print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    print("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print("Loading pretrained model from {}".format(pretrained_path))

    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class FaceDetector:
    """Класс для инференса нейронной сети для детекции человеческого лица.
    Основан на RetinaFace in PyTorch
    (https://github.com/biubug6/Pytorch_Retinaface)"""

    def __init__(self):
        # инициализируем RetinaFace с backbone Resnet50
        self.vis_thres = 0.5
        self.nms_threshold = 0.4
        self.confidence_threshold = 0.02

        # размер до которого уменьшаются все изображения
        self.max_size = 760

        # скачиваем готовую модель из Google Drive
        model_path = "./RetinaFace_weights/Resnet50_Final.pth"
        self.font = "./RetinaFace_weights/PTSans-Regular.ttf"

        if os.path.exists(model_path):
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            url = "https://drive.google.com/drive/folders/18719ky0AiBntVLxOTzRFk-oui3eefyRj"
            gdown.download_folder(url, quiet=False)
            self.model = torch.load(model_path, map_location=torch.device("cpu"))

        # net and model
        self.cfg = cfg_re50
        net = RetinaFace(cfg_re50, phase="test")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        load_to_cpu = self.device == "cpu"
        print("load_to_cpu", load_to_cpu)
        net = load_model(net, model_path, load_to_cpu)
        net.eval()
        cudnn.benchmark = True

        self.net = net.to(self.device)

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

    def detect_faces(self, img_raw):
        # подготавливаем картинку
        img, scale, resize = self.prepare_pil_img(img_raw, self.max_size)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)
        im_height, im_width = img.shape[2:]
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        dets = dets[dets[:, 4] > self.vis_thres]

        return dets

    @staticmethod
    def prepare_pil_img(img_raw, max_size=760):
        img = np.array(img_raw)  # .astype("float32")
        # переводим rgb в bgr
        img = img[:, :, ::-1]

        # уменьшаем размер картинки, если нужно
        im_shape = img.shape
        im_size_max = np.max(im_shape[0:2])
        # prevent bigger axis from being more than max_size:
        if np.round(im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        else:
            resize = 1

        if resize != 1:
            # img = cv2.resize(
            #     img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
            # )
            im_height, im_width, _ = img.shape
            new_size = (int(im_width * resize), int(im_height * resize))
            img = Image.fromarray(img, "RGB").resize(new_size, resample=Image.BILINEAR)
            img = np.array(img)

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = img.astype("float32")
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()).unsqueeze(0)

        return img, scale, resize

    def load_image_and_detect_faces(self, path2image):
        image_rgb = self.load_image(path2image)
        detections = self.detect_faces(image_rgb)

        return detections, image_rgb

    @staticmethod
    def viz_detections(image_rgb, detections):
        draw = ImageDraw.Draw(image_rgb)

        for det in detections:
            draw.rectangle(
                (det[0], det[1], det[2], det[3]), outline=(0, 255, 0), width=3
            )

        return image_rgb

    def viz_detections_by_data(self, image_rgb, detection, emotion_cls, score):
        draw = ImageDraw.Draw(image_rgb)

        face_bbox = (detection[0], detection[1], detection[2], detection[3])
        draw.rectangle(face_bbox, outline=(0, 255, 0), width=3)

        # Portion of the image the text width should be (between 0 and 1)
        width_ratio = 0.2

        text = f"{emotion_cls}:{score:.2%}"

        font_size = find_font_size(text, self.font, image_rgb, width_ratio)
        fnt = ImageFont.truetype(self.font, font_size)

        draw.text((detection[0], detection[1]), text, fill=(0, 255, 0), font=fnt)

        return image_rgb


class DetectEmotionOnFaces(FaceDetector):
    def __init__(self, emotion_net):
        super().__init__()

        self.emotion_net = emotion_net

    def viz_emo_detections(self, image_rgb, detections):
        draw = ImageDraw.Draw(image_rgb)

        for det in detections:
            face_bbox = (det[0], det[1], det[2], det[3])
            draw.rectangle(face_bbox, outline=(0, 255, 0), width=3)

            face_crop = image_rgb.crop(face_bbox)

            emotion_cls, score = self.emotion_net.predict_on_image(
                face_crop, return_label=True
            )

            width_ratio = (
                0.2  # Portion of the image the text width should be (between 0 and 1)
            )

            text = f"{emotion_cls}:{score:.2%}"
            font_size = find_font_size(text, self.font, image_rgb, width_ratio)
            fnt = ImageFont.truetype(self.font, font_size)

            draw.text((det[0], det[1]), text, fill=(0, 255, 0), font=fnt)

        return image_rgb
