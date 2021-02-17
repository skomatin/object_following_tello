from typing import Union, Optional, List, Tuple, Text, BinaryIO
import torchvision
import torchvision.transforms as T
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageColor
import time
# import matplotlib.pyplot as plt

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

PREDICTION_SCORE_CONFIDENCE_THRESHOLD = 0.7


def tic():
    return time.time()

def toc(tstart, name="Operation"):
    print('%s took: %s sec.\n' % (name, (time.time() - tstart)))


class ImgProc():
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    #Soruce: https://github.com/pytorch/vision/blob/master/torchvision/utils.py
    def draw_bounding_boxes(self,
            image: torch.Tensor,
            boxes: torch.Tensor,
            labels: Optional[List[str]] = None,
            colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
            fill: Optional[bool] = False,
            width: int = 1,
            font: Optional[str] = None,
            font_size: int = 10
    ) -> torch.Tensor:
        """
        Draws bounding boxes on given image.
        The values of the input image should be uint8 between 0 and 255.
        If filled, Resulting Tensor should be saved as PNG image.
        Args:
            image (Tensor): Tensor of shape (C x H x W)
            boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
                the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
                `0 <= ymin < ymax < H`.
            labels (List[str]): List containing the labels of bounding boxes.
            colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
                be represented as `str` or `Tuple[int, int, int]`.
            fill (bool): If `True` fills the bounding box with specified color.
            width (int): Width of bounding box.
            font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
                also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
                `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
            font_size (int): The requested font size in points.
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Tensor expected, got {type(image)}")
        elif image.dtype != torch.uint8:
            raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")

        ndarr = image.permute(1, 2, 0).numpy()
        img_to_draw = Image.fromarray(ndarr)

        img_boxes = boxes.to(torch.int64).tolist()

        if fill:
            draw = ImageDraw.Draw(img_to_draw, "RGBA")

        else:
            draw = ImageDraw.Draw(img_to_draw)

        txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

        for i, bbox in enumerate(img_boxes):
            if colors is None:
                color = None
            else:
                color = colors[i]

            if fill:
                if color is None:
                    fill_color = (255, 255, 255, 100)
                elif isinstance(color, str):
                    # This will automatically raise Error if rgb cannot be parsed.
                    fill_color = ImageColor.getrgb(color) + (100,)
                elif isinstance(color, tuple):
                    fill_color = color + (100,)
                draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
            else:
                draw.rectangle(bbox, width=width, outline=color)

            if labels is not None:
                draw.text((bbox[0], bbox[1]), labels[i], fill=color, font=txt_font)

        # return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)
        return np.array(img_to_draw)


    def detect_object(self, img, name='test'):
        #Resize image to speed up inference
        img = cv2.resize(img, (400, 400))

        #Convert image to PIL image
        ts = tic()
        # pil_img = Image.fromarray(img)
        pil_img = Image.fromarray(np.uint8(img))

        #Conovert PIL image to Tensor
        tensor_img = T.Compose([T.ToTensor()])(pil_img)

        #Run Inference on image
        ts_detect = tic()
        detections = self.model([tensor_img])
        toc(ts_detect, 'model inference')

        #Extract all confident predictions
        ts_conf = tic()
        detections_confident = {'boxes': [], 'labels': [], 'scores': []}
        for pred_idx in range(len(detections[0]['boxes'])):
            score = detections[0]['scores'][pred_idx]
            if score > PREDICTION_SCORE_CONFIDENCE_THRESHOLD:
                detections_confident['boxes'].append(detections[0]['boxes'][pred_idx, :])
                detections_confident['labels'].append(detections[0]['labels'][pred_idx])
                detections_confident['scores'].append(detections[0]['scores'][pred_idx])

        toc(ts_conf, 'getting confident boxes')

        #Get confident bounding boxes and labels
        boxes = torch.cat(detections_confident['boxes']).reshape(len(detections_confident['labels']), 4)
        labels = [COCO_INSTANCE_CATEGORY_NAMES[idx] for idx in detections_confident['labels']]

        #Draw bounding boxes on images
        ts_draw_bbox = tic()
        img_with_boxes = self.draw_bounding_boxes(tensor_img.mul(255).type(dtype=torch.uint8), boxes, labels)
        toc(ts_draw_bbox, "Drawing bounding boxes")

        #Save image with bounding boxes
        cv2.imwrite('images/{}.png'.format(name), cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))

        #Look for a water bottle
        bbox = None
        for object_id in range(len(detections_confident['labels'])):
            if COCO_INSTANCE_CATEGORY_NAMES[detections_confident['labels'][object_id]] == "bottle":
                bbox = detections_confident['boxes'][object_id]
                break

        if bbox is not None:
            print("Water Bottle found!\nCoordinates: = ", bbox)
            # cv2.imwrite('images/bottle_{}.png'.format(name), cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))

        else:
            print("No water bottle found")

        toc(ts, 'Object Detection')

if __name__ == "__main__":
    # Read image
    img = Image.open('sample_img.png')

    img_proc = ImgProc()
    img = img_proc.detect_object(np.array(img), 'sample')
