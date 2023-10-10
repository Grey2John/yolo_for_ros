#!/home/zlh/anaconda3/envs/py_yolo/lib python3.8

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
from rostopic import get_topic_type

sys.path.insert(0, "/home/zlh/ws/deep_learning/YOLO_train/yolo_set/ws/yolo/src/yolov5_ros-main/scripts")

from sensor_msgs.msg import Image, CompressedImage
from yolov5_ros.msg import BoundingBox, BoundingBoxes
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.augmentations import letterbox


# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


# @torch.no_grad()
@smart_inference_mode()
class Yolov5Segment:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        # self.max_det = rospy.get_param("~maximum_detections")
        self.retina_masks = rospy.get_param("~retina_masks", True)
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        
        self.hide_labels = rospy.get_param("~hide_labels", False)
        self.hide_conf = rospy.get_param("~hide_conf", False)
        self.half = rospy.get_param("~half", False) 
        # Initialize weights 
        weights = rospy.get_param("~weights")
        # load model
        self.device = select_device(str(rospy.get_param("~device",""))) # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"), fp16=self.half)
        self.stride, self.names, self.pt, = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            # self.model.jit,
            # self.model.onnx,
            # self.model.engine,  #  ??????
        )
        # Half use FP16 half-precision inference
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_h", 640), rospy.get_param("~inference_size_w",640)]
        self.img_size = check_img_size(self.img_size, s=self.stride)
        
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_size))  # warmup        
        
        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"  #  compress image msgs
        # rospy.loginfo("the image message type is: {}".format(input_image_type))

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.PredictCallback, queue_size=300
            )  # sub image and run callback function, CompressedImage is the type of message
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.PredictCallback, queue_size=300
            )
        # Initialize image/box publisher
        self.if_publish_image = rospy.get_param("~if_publish_image")
        self.if_publish_box = rospy.get_param("~if_publish_box")
        if self.if_publish_image:
            self.pred_image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), CompressedImage, queue_size=30
            )
        if self.if_publish_box:
            self.pred_box_pub = rospy.Publisher(
                rospy.get_param("~output_box_topic"), BoundingBoxes, queue_size=30
            )
        # Initialize CV_Bridge
        self.bridge = CvBridge()

        self.cut_start = int((1280 - self.img_size[1])/2)
        self.cut_end = self.cut_start + 960


    def PredictCallback(self, data):
        """adapted from yolov5/segment.py"""
        # load data -- image
        if self.compressed_input:
            img = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        # rospy.loginfo("get the image from cv2 {}".format(img.shape))  # (720, 1280, 3)
        # cut the image from 1280x720 to 960x720
        im, im0 = self.preprocess(img)
        # rospy.loginfo("get the iamge, the im shape is {}".format(im.shape)) # im shape is (1, 3, 736, 960)
        # rospy.loginfo("get the iamge, the im0 shape is {}".format(im0.shape)) # im0 shape is (720, 960, 3)

        # Run inference
        seen, dt = 0, (Profile(), Profile(), Profile())
        with dt[0]:
            im = torch.from_numpy(im).to(self.device)  # send pic to GPU
            im = im.half() if self.half else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
        # Inference
        with dt[1]:
            pred, proto = self.model(im, augment=False, visualize=False)[:2]
            print("the predict result shape is: {}".format(pred.shape)) # ([1, 43470, 43])
            print("the predict result proto is: {}".format(proto.shape)) # ([1, 32, 184, 240])
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                       self.classes, self.agnostic_nms, nm=32)

        # Process predictions -- BB and mask
        ### To-do move pred to CPU and fill BoundingBox messages
        for i, det in enumerate(pred):  # per image
            seen += 1
            # det = pred[0].cpu().numpy()

            bounding_boxes = BoundingBoxes() # class init
            bounding_boxes.header = data.header
            bounding_boxes.image_header = data.header
            
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                if self.retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
                    255 if self.retina_masks else im[i])
                    
                # bouding box
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    bounding_box = BoundingBox()
                    c = int(cls)
                    # Fill in bounding box message
                    bounding_box.Class = self.names[c]
                    bounding_box.probability = conf 
                    bounding_box.xmin = int(xyxy[0])
                    bounding_box.ymin = int(xyxy[1])
                    bounding_box.xmax = int(xyxy[2])
                    bounding_box.ymax = int(xyxy[3])

                    bounding_boxes.bounding_boxes.append(bounding_box)

                    # Annotate the image
                    if self.if_publish_image or self.if_publish_box:  # Add bbox to image
                            # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))       
                    
                    ### POPULATE THE DETECTION MESSAGE HERE

        # Stream results
        im0 = annotator.result()  # add annotation results to im0

        # Publish prediction
        if self.if_publish_box:
            self.pred_box_pub.publish(bounding_boxes)
        if self.if_publish_image:
            self.pred_image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(im0))
        

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        # rospy.loginfo(im.shape)
        # rospy.loginfo("the image type is: {}".format(type(img)))
        img=img[:, self.cut_start:self.cut_end, :]  # HWC
        img_origin = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]]) 
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img_origin

    # def post_process(self, ):



if __name__ == "__main__":
    # check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5_seg", anonymous=True)
    detector = Yolov5Segment()
    
    rospy.spin()
