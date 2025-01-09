from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2


class Perception(torch.nn.Module):
    YOLO_CLASS_MAP = {
        "person":           0,
        "bicycle":          1,
        "car":              2,
        "motorcycle":       3,
        "airplane":         4,
        "bus":              5,
        "train":            6,
        "truck":            7,
        "boat":             8,
        "traffic light":    9,
        "fire hydrant":     10,
        "stop sign":        11,
        "parking meter":    12,
        "bench":            13,
        "bird":             14,
        "cat":              15,
        "dog":              16,
        "horse":            17,
        "sheep":            18,
        "cow":              19,
        "elephant":         20,
        "bear":             21,
        "zebra":            22,
        "giraffe":          23,
        "backpack":         24,
        "umbrella":         25,
        "handbag":          26,
        "tie":              27,
        "suitcase":         28,
        "frisbee":          29,
        "skis":             30,
        "snowboard":        31,
        "sports ball":      32,
        "kite":             33,
        "baseball bat":     34,
        "baseball glove":   35,
        "skateboard":       36,
        "surfboard":        37,
        "tennis racket":    38,
        "bottle":           39,
        "wine glass":       40,
        "cup":              41,
        "fork":             42,
        "knife":            43,
        "spoon":            44,
        "bowl":             45,
        "banana":           46,
        "apple":            47,
        "sandwich":         48,
        "orange":           49,
        "brocolli":         50,
        "carrot":           51,
        "hot dog":          52,
        "pizza":            53,
        "donut":            54,
        "cake":             55,
        "chair":            56,
        "couch":            57,
        "potted plant":     58,
        "bed":              59,
        "dining table":     60,
        "toilet":           61,
        "tv":               62,
        "laptop":           63,
        "mouse":            64,
        "remote":           65,
        "keyboard":         66,
        "cell phone":       67,
        "microwave":        68,
        "oven":             69,
        "toaster":          70,
        "sink":             71,
        "refrigerator":     72,
        "book":             73,
        "clock":            74,
        "vase":             75,
        "scissors":         76,
        "teddy bear":       77,
        "hair drier":       78,
        "toothbrush":       79
    }

    def __init__(self, 
                 depth_model: str = "vitl",
                 depth_input_size: int = 518,
                 yolo_model: str = "yolo11l",
                 device: str = "cuda",
                 side_ratio: float = 0.25,
                 safe_distance: float = 0.5,
                 beta: float = 0.6,
                 visualize: bool = False):
        super().__init__()

        self.depth_input_size = depth_input_size
        self.side_ratio = side_ratio
        self.safe_distance = safe_distance
        self.beta = beta
        self.visualize = visualize
        self.depth_anything = self.load_model(depth_model, device)
        self.yolo_model = YOLO(f"./checkpoints/{yolo_model}.pt")

        if self.visualize:
            # visualization stuff
            self.cmap = matplotlib.colormaps.get_cmap("Spectral_r")
            cv2.namedWindow("Depth Anything V2", cv2.WINDOW_NORMAL)
            cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)

    def load_model(self, encoder, device) -> DepthAnythingV2:
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]}
        }
        # depth_anything = DepthAnythingV2(**model_configs[encoder])
        # depth_anything.load_state_dict(torch.load(f"../checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu"))
        depth_anything = DepthAnythingV2(**{**model_configs[encoder], "max_depth": 10})
        depth_anything.load_state_dict(torch.load(f"./checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth", map_location="cpu"))
        depth_anything = depth_anything.to(device).eval()

        return depth_anything


    def get_depth_image(self, raw_frame: np.ndarray) -> torch.Tensor:
        # depth = depth_anything.infer_image(raw_frame, input_size)
        image, (h, w) = self.depth_anything.image2tensor(raw_frame, self.depth_input_size)
        depth: torch.Tensor = self.depth_anything.forward(image)
        depth = torch.nn.functional.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

        return depth

    def get_depth_image_visual(self, depth_frame: torch.Tensor, grayscale: bool = False) -> np.ndarray:
        # create a copy of the depth tensor on cpu
        depth_img = depth_frame.detach().cpu().numpy()
        
        if grayscale:
            depth_img = depth_img * 100.0
            depth_img = np.clip(depth_img, 0, 255)
            depth_img = depth_img.astype(np.uint8)
            depth_img = np.repeat(depth_img[..., np.newaxis], 3, axis=-1)

        else:
            depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
            depth_img = (self.cmap(depth_img)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        return depth_img

    def avoid_obstacles_planner(self, depth_frame: torch.Tensor) -> torch.Tensor:
        height, width = depth_frame.shape[:2]

        side_width = int(width * self.side_ratio)
        
        average_distances = torch.zeros(3)
        min_distances = torch.zeros(3)

        # left side
        left_side = depth_frame[:, :side_width]
        average_distances[0] = torch.mean(left_side)
        min_distances[0] = torch.min(left_side)
        # right side
        right_side = depth_frame[:, width - side_width:]
        average_distances[2] = torch.mean(right_side)
        min_distances[2] = torch.min(right_side)
        
        # center
        center = depth_frame[:, side_width:width - side_width]
        average_distances[1] = torch.mean(center)
        min_distances[1] = torch.min(center)

        thresholds = (self.beta * min_distances + (1-self.beta) * average_distances) < self.safe_distance

        return thresholds

    def show_debug_view(self, depth_frame: torch.Tensor, yolo_results: list[Any], thresholds: torch.Tensor):
        depth_img = self.get_depth_image_visual(depth_frame)

        # draw lines separating the slices
        side_width = int(depth_img.shape[1] * self.side_ratio)
        cv2.line(depth_img, (side_width, 0), (side_width, depth_img.shape[0]), (255, 255, 255), 2)
        cv2.line(depth_img, (depth_img.shape[1] - side_width, 0), (depth_img.shape[1] - side_width, depth_img.shape[0]), (255, 255, 255), 2)
        
        # highlight the slice with the lowest average distance
        if thresholds[0]:
            cv2.rectangle(depth_img, (0, 0), (side_width, depth_img.shape[0]), (0, 0, 255), 2)
        if thresholds[1]:
            cv2.rectangle(depth_img, (side_width, 0), (depth_img.shape[1] - side_width, depth_img.shape[0]), (0, 0, 255), 2)
        if thresholds[2]:
            cv2.rectangle(depth_img, (depth_img.shape[1] - side_width, 0), (depth_img.shape[1], depth_img.shape[0]), (0, 0, 255), 2)

        annotated_frame = yolo_results[0].plot()

        cv2.imshow("Depth Anything V2", depth_img)
        cv2.imshow("YOLO", annotated_frame)
        cv2.waitKey(1)


    def forward(self, raw_frame: np.ndarray) -> torch.Tensor:        
        # depth perception
        depth_frame = self.get_depth_image(raw_frame)

        # object detection
        yolo_results = self.yolo_model.predict(raw_frame, conf=0.2, classes=[self.YOLO_CLASS_MAP["banana"]], verbose=False)

        # obstacle avoid based on depth
        thresholds = self.avoid_obstacles_planner(depth_frame)

        if self.visualize:
            self.show_debug_view(depth_frame, yolo_results, thresholds)
        
        goal = torch.zeros(3)

        # do path planning
        if thresholds[0]:
            # move to the right
            goal[1] = -0.2
        elif thresholds[2]:
            # move to the left
            goal[1] = 0.2
        else:
            goal[1] = 0

        if thresholds[1]:
            # move forward
            goal[0] = 0
        else:
            goal[0] = 0.2
            

        if yolo_results:
            boxes = yolo_results[0].boxes

            if boxes:
                assert isinstance(boxes.conf, torch.Tensor)
                
                # get the highest probability class
                highest_idx = torch.argmax(boxes.conf)
                xywhn = boxes[highest_idx].xywhn

                goal[2] = xywhn[0, 0] - 0.5

        return goal



if __name__ == "__main__":
    depth_model = "vits"
    depth_input_size = 518
    yolo_model = "yolo11l"
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # create the model
    model = Perception(depth_model=depth_model, depth_input_size=depth_input_size, yolo_model=yolo_model, device=device, visualize=True)

    # open the video source
    video_source = 0
    raw_video = cv2.VideoCapture(video_source)
    if not raw_video.isOpened():
        print("Error: Could not open video source.")
        exit()
    
    
    while True:  # Loop for a maximum number of frames
        ret, raw_frame = raw_video.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        

        goal = model.forward(raw_frame)

        print(goal)

    
    raw_video.release()
    cv2.destroyAllWindows()