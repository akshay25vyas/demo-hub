import numpy as np
import cv2
import argparse

import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray

from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_matrix


from _init_paths import *
from lib.hl2ss import hl2ss_3dcv, hl2ss
from lib.HoloDevicePortal import HoloDevicePortal


import os
import cv2
import torch
import numpy as np
from absl import app
from absl import flags
from PIL import Image as PILImg
import matplotlib.pyplot as plt
from utils.obj_detection import GroundingDINO
from groundingdino.util.inference import annotate
from utils.helpers import (P, show_mask, load, get_clip)
from torchvision.ops import box_convert


# Define flags
flags.DEFINE_string('dataset_path', '/home/user/Datasets/ptg-november23demo/Object_Detection', 'Path to the dataset folder')
flags.DEFINE_float('box_threshold', 0.5, 'Box threshold for GroundingDINO')
flags.DEFINE_float('text_threshold', 0.5, 'Text threshold for GroundingDINO')
flags.DEFINE_float('alpha', 0.5, 'alpha parameter for Proto-CLIP')
flags.DEFINE_float('beta', 0.15, 'beta paramater for Proto-CLIP')
flags.DEFINE_float('pct', 0.2, 'Threshold for Proto-CLIP')
FLAGS = flags.FLAGS


def process_bbox(boxes, h, w):
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.int32)
    return xyxy


class DetectionPublisher:
    def __init__(
        self, device_name, pv_res="720P", depth_mode="LONGTHROW",device="cuda:0", debug=False
    ) -> None:
        if not self._is_roscore_running():
            raise RuntimeError("roscore is not running")

        self._init_node("image_listener", debug)
        self._bridge = CvBridge()

        if pv_res == "360P":
            self._pv_width = 640
            self._pv_height = 360
        if pv_res == "720P":
            self._pv_width = 1280
            self._pv_height = 720
        if pv_res == "1080P":
            self._pv_width = 1920
            self._pv_height = 1080

        self._depth_mode = depth_mode
        if depth_mode == "AHAT":
            self._depth_width = hl2ss.Parameters_RM_DEPTH_AHAT.WIDTH
            self._depth_height = hl2ss.Parameters_RM_DEPTH_AHAT.HEIGHT
        if depth_mode == "LONGTHROW":
            self._depth_width = hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH
            self._depth_height = hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT

        self._calib_folder = PROJ_ROOT / "config/calibration" / device_name
        self._pv_calibration = self._load_pv_calibration()
        self._depth_calibration = self._load_depth_calibration()

        self._topic_list = [
            "/hololens/pv_camera/image_raw",
            "/hololens/pv_camera/pose",
            "/hololens/depth_camera/image_raw",
            "/hololens/depth_camera/pose",
        ]
        self._prev_stamp = -1
        self.synced_msgs = []

        self._modelPub = rospy.Publisher(
            f"/model/object_location", Float32MultiArray, queue_size=2
        )

        self._ptgPub = rospy.Publisher(
            f"/ptg/object", String, queue_size=2
        )


        self._aligned_pv_frame_id = "hololens_aligned_pv_optical_frame"

        self._device = device
        self._z_img_proto = torch.load(os.path.join(os.getcwd(), 'pretrained_checkpoints', 'proto-clip', 'image_features.pt')).to(device) # (NxD)
        self._z_txt_proto = torch.load(os.path.join(os.getcwd(), 'pretrained_checkpoints', 'proto-clip', 'text_features.pt')).to(device) # (NxD)
        self._class_names = sorted(load(os.path.join(os.getcwd(), 'pretrained_checkpoints', 'proto-clip', 'class_folder_mapping.pkl'), 'class_names').keys())
        self._gdino = GroundingDINO(
        './utils/submodules/GroundingDINO',
        './pretrained_checkpoints/gdino/groundingdino_swint_ogc.pth',
        FLAGS.box_threshold,
        FLAGS.text_threshold)

        self._model, preprocess = get_clip(self._device)





    def run(self):
        def process_synced_msgs(msgs):
            color = self._imgmsg_to_cv2(msgs[0])[:,:,::-1].copy()
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            bboxes, logits, class_labels = self._gdino.detect(color)
            bounding_boxes = process_bbox(bboxes, self._pv_height, self._pv_width)
            zq_imgs_flat = []
            with torch.no_grad():
                for box in bounding_boxes:
                    x_min, y_min, x_max, y_max = box
                    # Crop the bounding box from the image
                    cropped_img = image[y_min:y_max, x_min:x_max]
                    crop_img_tensor = preprocess(PILImg.fromarray(cropped_img)).unsqueeze(0).to(self._device)
                    image_features = model.encode_image(crop_img_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    zq_imgs_flat.append(image_features)
            zq = torch.cat(zq_imgs_flat, dim=0)
            p = P(zq, z_img_proto, z_txt_proto, alpha=FLAGS.alpha, beta=FLAGS.beta)
            max_confs, max_idx = torch.max(p, dim=1)
            cls = [class_names[idx] for idx in max_idx]
            filtered_idxs = [ idx for idx, conf in enumerate(max_confs) if conf > FLAGS.pct]
            if len(filtered_idxs) > 0:
                _bboxes = bboxes[filtered_idxs]
                _max_confs = [ max_confs[idx] for idx in filtered_idxs ]
                _cls = [ cls[idx] for idx in filtered_idxs ]
                annotated_frame = annotate(image_source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), boxes=_bboxes, logits=_max_confs, phrases=_cls)
                input_boxes, masks = gdino.get_sam_mask(image, _bboxes)



                depth = self._imgmsg_to_cv2(msgs[2])
                pv2world = self._posemsg_to_numpy(msgs[1])
                depth2world = self._posemsg_to_numpy(msgs[3])
                depth2pv = depth2world @ np.linalg.inv(pv2world)
                depth2pv = depth2pv.T

                depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
                points = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
                pv_uv = hl2ss_3dcv.project(points, depth2pv @ color_intrinsics)
                
                # TODO
                ptg_result =[]


                # gpu to parallelize to reduce inference time.
                for mask in masks:
                    mask = mask.cpu().numpy().astype(np.uint8)
                    aligned_mask = cv2.remap(mask, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)
                    obj_pts = points[aligned_mask]
                    center = np.mean(obj_pts, axis=0)
                    cls_name= "xxxxxx"
                    conf = 0.99
                    ptg_result.append(
                        f"{cls_name}|{center[0]}|{center[1]}|{center[2]}|{conf}"
                    )

                    

                timestamp = rospy.Time.now()
                ptgMsg = self._createPtgMsg(ptg_result, timestamp)
                self._ptgPub.Publish(ptgMsg)
            
            ## TODO
            arry_for_model = xxxxxx
            timestamp = rospy.Time.now()
            modelMsg = self._create_modelMsg(arry_for_model, timestamp)
            self._modelPub.Publish(modelMsg)

        def callback(*msgs):
            if msgs:
                stamp = msgs[0].header.stamp.to_nsec()
                if stamp != self._prev_stamp:
                    process_synced_msgs(msgs)
                    self._prev_stamp = stamp
            else:
                rospy.logwarn("No synced messages received!!!")

        # alignedpvPub = rospy.Publisher(
        #     f"/hololens/pv_aligned_to_depth/image_raw", Image, queue_size=2
        
        

        pv_extrinsics = np.eye(4, 4, dtype=np.float32)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(
            self._pv_calibration.intrinsics, pv_extrinsics
        )

        uv2xy = hl2ss_3dcv.compute_uv2xy(
            self._depth_calibration.intrinsics, self._depth_width, self._depth_height
        )
        xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(
            uv2xy, self._depth_calibration.scale
        )
        undistort_map = self._depth_calibration.undistort_map
        rig2depth = self._depth_calibration.extrinsics
        depth2rig = np.linalg.inv(rig2depth)

        self._sub_list = self._create_sub_list()
        self._ts = ApproximateTimeSynchronizer(self._sub_list, 2, slop=0.1)
        self._ts.registerCallback(callback)

    def _load_pv_calibration(self):
        calib_folder = (
            self._calib_folder
            / f"personal_video/1000_{self._pv_width}_{self._pv_height}"
        )
        return hl2ss_3dcv._load_calibration_pv(str(calib_folder))

    def _load_depth_calibration(self):
        if self._depth_mode == "AHAT":
            return hl2ss_3dcv._load_calibration_rm_depth_ahat(
                str(self._calib_folder / f"rm_depth_{self._depth_mode.lower()}")
            )
        if self._depth_mode == "LONGTHROW":
            return hl2ss_3dcv._load_calibration_rm_depth_longthrow(
                str(self._calib_folder / f"rm_depth_{self._depth_mode.lower()}")
            )

    # TODO::
    def _createPtgMsg(self, result_lst, timestamp):
        pass

    # TODO
    def _create_modelMsg(self, model_arry, timestamp):
        pass

    def _create_msgImage(self, image_array, timestamp, frame_id):
        """
        ref: http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html
        ref: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        :param image_array
            numpy array of the image
        :param encoding
            encoding type of the image ()

        :returns msgImage
        """
        msg = self._bridge.cv2_to_imgmsg(image_array)
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        return msg

    def _create_sub_list(self):
        sub_list = []
        for topic in self._topic_list:
            if "image_raw" in topic:
                sub_list.append(Subscriber(topic, Image, queue_size=10))
            if "pose" in topic:
                sub_list.append(Subscriber(topic, PoseStamped, queue_size=10))
        return sub_list

    def _is_roscore_running(self):
        try:
            rospy.get_master().getPid()
            return True
        except:
            return False

    def _init_node(self, node_id, debug):
        if not self._is_roscore_running():
            raise RuntimeError("roscore is not running")
        rospy.init_node(
            node_id, anonymous=True, log_level=rospy.DEBUG if debug else rospy.INFO
        )

    def stop(self):
        for sub in self._sub_list:
            sub.unregister()
        rospy.signal_shutdown("user requested...")

    def _imgmsg_to_cv2(self, img_msg):
        """Convert sensor_msgs.Image to numpy.ndarray.

        Args:
            img_msg (sensor_msgs.Image): Image message.

        Returns:
            np.ndarray: Image as numpy array.
        """
        if img_msg.encoding == "16UC1":
            dtype, n_channels = np.uint16, 1
        elif img_msg.encoding == "8UC3":
            dtype, n_channels = np.uint8, 3
        elif img_msg.encoding == "bgr8":
            dtype, n_channels = np.uint8, 3
        elif img_msg.encoding == "rgb8":
            dtype, n_channels = np.uint8, 3
        elif img_msg.encoding == "bgra8":
            dtype, n_channels = np.uint8, 4
        elif img_msg.encoding == "rgba8":
            dtype, n_channels = np.uint8, 4
        else:
            raise ValueError(f"Unknown encoding: {img_msg.encoding}")

        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder(">" if img_msg.is_bigendian else "<")
        tmp_data = img_msg.data

        if n_channels == 1:
            im = np.ndarray(
                shape=(img_msg.height, img_msg.width), dtype=dtype, buffer=tmp_data
            )
        else:
            im = np.ndarray(
                shape=(img_msg.height, img_msg.width, n_channels),
                dtype=dtype,
                buffer=tmp_data,
            )

        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == "little"):
            im = im.byteswap().newbyteorder()

        if img_msg.encoding == "rgb8":
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        if img_msg.encoding == "rgba8":
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        if img_msg.encoding == "bgra8":
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

        return im

    def _posemsg_to_numpy(self, pose_msg):
        quaternion = (
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        )
        pose = quaternion_matrix(quaternion)
        pose[0:3, 3] = [
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
        ]
        return pose.astype(np.float32)


def args_parser():
    parser = argparse.ArgumentParser(description="Aligned Color to Depth Publisher")
    parser.add_argument(
        "--host", type=str, default="192.168.50.100", help="HoloLens 2 IP address"
    )
    parser.add_argument(
        "--username", type=str, default="admin", help="HoloLens 2 username"
    )
    parser.add_argument(
        "--password", type=str, default="123456789", help="HoloLens 2 password"
    )
    parser.add_argument(
        "--pv_res",
        default="720P",
        choices=["360P", "720P", "1080P"],
        help="Photo Video (PV) camera resolution",
    )
    parser.add_argument(
        "--depth_mode",
        default="LONGTHROW",
        choices=["AHAT", "LONGTHROW"],
        help="Depth camera mode",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()

    portal = HoloDevicePortal(args.host, args.username, args.password)

    # get device name
    device_name = portal.get_machine_name().lower().replace(" ", "_").replace("-", "_")

    listener = AlignedImagesListener(
        device_name, args.pv_res, args.depth_mode, args.debug
    )
    listener.run()
    rospy.spin()
    listener.stop()