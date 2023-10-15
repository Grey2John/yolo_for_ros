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
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.segment.general import masks2segments, process_mask, process_mask_grey
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
        self.retina_masks = rospy.get_param("~retina_masks", True) # high resolution
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
        self.img_size = [rospy.get_param("~inference_size_h", 640), rospy.get_param("~inference_size_w_cut",640)]
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
        self.if_publish_mask = rospy.get_param("~if_publish_mask")
        self.if_publish_rgb8 = rospy.get_param("~if_publish_rgb8")
        self.if_publish_box = rospy.get_param("~if_publish_box")
        if self.if_publish_rgb8:
            self.origin_w = rospy.get_param("~inference_size_w")
            self.pred_rgba8_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic")+"/bgra", Image, queue_size=30
            )
        if self.if_publish_mask:
            self.pred_image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic")+"mono", Image, queue_size=30
            )
        # Initialize CV_Bridge
        self.bridge = CvBridge()

        self.cut_start = int((1280 - self.img_size[1])/2)
        self.cut_end = self.cut_start + 960


    def PredictCallback(self, data):
        """adapted from yolov5/segment.py"""
        # print(data.header.stamp.nsecs)
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
            # print("the predict result shape is: {}".format(pred.shape)) # ([1, 43470, 43])
            # print("the predict result proto is: {}".format(proto.shape)) # ([1, 32, 184, 240])
        # NMS 
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                       self.classes, self.agnostic_nms, nm=32)

        # Process predictions -- BB and mask
        ### To-do move pred to CPU and fill BoundingBox messages
        masks = torch.zeros([1, 720, 960])
        for i, det in enumerate(pred):  # per image
            seen += 1
            # det = pred[0].cpu().numpy()
            if len(det):
                if self.retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size [4, 38]
                    masks = process_mask_grey(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # CHW [n, 720, 960]
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Stream results
        # im0 = annotator.result()  # add annotation results to im0
        if self.if_publish_rgb8:
            mix_msg = self.mix_post_process(masks, det[:, 5], data, img)
            self.pred_rgba8_pub.publish(mix_msg)
        if self.if_publish_mask:
            grey_msg = self.mask_post_process(masks, det[:, 5], data)
            # print(grey_msg.header.stamp.nsecs)
            self.pred_image_pub.publish(grey_msg)
        
        
    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class. 
        generate the inference sample typically
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

    def mask_post_process(self, masks, class_num, data):
        masks = masks.detach().cpu().numpy()
        class_num = class_num.detach().cpu().numpy()
        # print(class_num)
        indexs = []
        for l in range(0, 2):
            indexs.append([index for index, value in enumerate(class_num) if value == l])

        if len(indexs[0]) < 2 and len(indexs[1]) < 2:
            masks_ = masks
        else:
            masks_ = np.zeros((2, 720, 960))
            class_list = []
            for label in range(0, 2):
                if len(indexs[label]) != 0:
                    class_list.append(label)
                    masks_[label] = np_merge(masks, indexs[label])
            class_num = np.array(class_list)

        grey_mask = np.zeros((720, 960), dtype=np.uint8)
        for i, c in enumerate(class_num):
            grey_mask += (masks_[i]*int(c+1)).astype(np.uint8)
        grey_mask_ = np.where(grey_mask>2, 100, grey_mask*50)

        grey_msg = self.bridge.cv2_to_imgmsg(grey_mask_, encoding='mono8')
        # grey_msg = self.bridge.cv2_to_compressed_imgmsg(grey_mask, encoding='mono8')
        grey_msg.header = data.header
        grey_msg.header.frame_id = "mask_frame"
        return grey_msg

    def mix_post_process(self, masks, class_num, data, origin_img):
            masks = masks.detach().cpu().numpy()
            class_num = class_num.detach().cpu().numpy()
            indexs = []
            for l in range(0, 2):
                indexs.append([index for index, value in enumerate(class_num) if value == l])

            if len(indexs[0]) < 2 and len(indexs[1]) < 2:
                masks_ = masks
            else:
                masks_ = np.zeros((2, 720, 960))
                class_list = []
                for label in range(0, 2):
                    if len(indexs[label]) != 0:
                        class_list.append(label)
                        masks_[label] = np_merge(masks, indexs[label])
                class_num = np.array(class_list)

            grey_mask = np.zeros((720, 960), dtype=np.uint8)
            for i, c in enumerate(class_num):
                grey_mask += (masks_[i]*int(c+1)).astype(np.uint8)
            grey_mask_ = np.where(grey_mask>2, 2, grey_mask)

            masks_big_size = np.zeros( (720, self.cut_start) )
            grey_mask_1 = np.concatenate((masks_big_size, grey_mask_), axis=1)
            grey_mask_2 = np.concatenate((grey_mask_1, masks_big_size), axis=1)
            mix_img = np.concatenate((origin_img, np.expand_dims(grey_mask_2, axis=2)), axis=2)  # (720, 1280, 4)
            
            mix_msg = self.bridge.cv2_to_imgmsg(mix_img.astype(np.uint8), encoding="bgra8")
            mix_msg.header = data.header
            mix_msg.header.frame_id = "mask_frame"
            return mix_msg


def np_merge(father_m, index):
    sub_matrix = np.take(father_m, index, axis=0)
    sum_matrix = np.sum(sub_matrix, axis=0)
    one_matrix = np.where(sum_matrix>0.5, 1, 0)
    return one_matrix


if __name__ == "__main__":
    # check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5_seg", anonymous=True)
    detector = Yolov5Segment()
    
    rospy.spin()
