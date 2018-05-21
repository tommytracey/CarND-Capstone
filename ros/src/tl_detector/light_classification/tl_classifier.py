from styx_msgs.msg import TrafficLight
import rospy

import tensorflow as tf
import os
import cv2
import numpy as np
from collections import Counter
import glob

# TODO(saajan): Remove this testing code
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TLClassifier(object):
    def __init__(self, vanilla_model_path, confidence_threshold, is_real, skip_frames):

        # TODO(saajan): Remove this testing code
        # self.cropped_boxes = rospy.Publisher('/cropped_boxes', Image, queue_size=20)
        # self.bridge = CvBridge()


        self.is_real = is_real
        self.skip_frames = skip_frames
        self.frames_count = 1
        self.confidence_threshold = confidence_threshold

        self.vanilla_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.vanilla_graph.as_default():
            graph_def = tf.GraphDef()

            rospy.loginfo("[TL Detector] Reading vanilla model: {0}".format(vanilla_model_path))

            with tf.gfile.GFile(vanilla_model_path, 'rb') as graph_file:
                read_graph_file = graph_file.read()

                graph_def.ParseFromString(read_graph_file)
                tf.import_graph_def(graph_def, name='')

            self.vanilla_sess = tf.Session(graph=self.vanilla_graph, config=config)

            self.vanilla_image_tensor = self.vanilla_graph.get_tensor_by_name('image_tensor:0')
            self.vanilla_num_detections =self.vanilla_graph.get_tensor_by_name('num_detections:0')
            # For each detection, it's corresponding bounding box, class and score:
            self.vanilla_boxes = self.vanilla_graph.get_tensor_by_name('detection_boxes:0')
            self.vanilla_classes = self.vanilla_graph.get_tensor_by_name('detection_classes:0')
            self.vanilla_scores =self.vanilla_graph.get_tensor_by_name('detection_scores:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(self.frames_count <= self.skip_frames):
            self.frames_count += 1

            rospy.loginfo("[TL Detector] Skipping detection for this frame...")
            return TrafficLight.SKIP
        else:
            self.frames_count = 1 #reset


        cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cropped_tls = self.get_boxes(cv2_image)

        if not self.is_real:
            # Simulator:
            # In the simulator, we should just check for the count of red pixels to detect red lights
            for cropped_tl in cropped_tls:
                # Detect number of red pixels
                cropped_image_hsv = cv2.cvtColor(cropped_tl, cv2.COLOR_RGB2HSV)

                # lower mask (0-10)
                lower_red_1 = np.array([0, 50, 50], dtype='uint8')
                upper_red_1 = np.array([10, 255, 255], dtype='uint8')

                # upper mask (170-180)
                lower_red_2 = np.array([170, 50, 50], dtype='uint8')
                upper_red_2 = np.array([180, 255, 255], dtype='uint8')

                red_mask_1 = cv2.inRange(cropped_image_hsv, lower_red_1, upper_red_1)
                red_mask_2 = cv2.inRange(cropped_image_hsv, lower_red_2, upper_red_2)

                combined_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

                red_pixels_count = cv2.countNonZero(combined_mask)

                #rospy.loginfo("[TL Detector] Simulator Red TL num pixels: " + str(red_pixels_count))

                if red_pixels_count > 150: # typical value is usually around 200
                    rospy.loginfo("[TL Detector] Detected RED light")
                    return TrafficLight.RED
                else:
                    return TrafficLight.UNKNOWN
        else:
            # Real:
            # Get the state of the detected traffic light using a classifier
            for cropped_tl in cropped_tls:
                #return self.get_state(cropped_tl)

                # Detect number of red pixels
                cropped_image_hsv = cv2.cvtColor(cropped_tl, cv2.COLOR_RGB2HSV)

                v_channel = cropped_image_hsv[:, :, 2]
                brightness = np.sum(v_channel)
                #rospy.loginfo("[TL Detector] Red TL brightness: " + str(brightness))

                if int(brightness) > 250000: # typical value is usually around 255000
                    rospy.loginfo("[TL Detector] Detected RED light")
                    return TrafficLight.RED
                else:
                    return TrafficLight.UNKNOWN

        # Fallback
        rospy.loginfo("[TL Detector] No TL detected...")
        return TrafficLight.UNKNOWN


    def get_boxes(self, image):
        with self.vanilla_graph.as_default():
            cv2_image_expanded = np.expand_dims(image, axis=0)

            (classes, scores, boxes) = self.vanilla_sess.run(
                [self.vanilla_classes, self.vanilla_scores, self.vanilla_boxes,],
                feed_dict={self.vanilla_image_tensor: cv2_image_expanded})
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)
            boxes = np.squeeze(boxes)

            tl_boxes = []
            for detected_class, score, box in zip(classes, scores, boxes):
                # Get TL boxes which have been detected with high enough confidence
                if score > self.confidence_threshold and detected_class == 10:
                    rospy.loginfo("[TL Detector] Detected TL | score: " + str(score))

                    box_in_pixels = self.normalized_box_to_pixels(box, image)

                    # Check for anomalies in box
                    # Box too small, skip
                    box_height = box_in_pixels[2] - box_in_pixels[0]
                    box_width = box_in_pixels[3] - box_in_pixels[1]
                    if(box_height < 15) or (box_width < 15):
                        continue
                    # Box can't be a TL due to H:W ratio, skip
                    if (box_height/box_width < 1.5):
                        continue

                    tl_boxes.append(box_in_pixels)
                    break # Return just one highest confidence box

            # Extract cropped TLs using boxes
            # Extract only the top portion of the detected TL (which will contain the red light) -
            #   hence the padding in resize below
            cropped_tls = []
            #image_as_np_array = np.asarray(image, dtype="uint8")
            #rospy.loginfo("[TL Detector] Image size: " +str(image.shape))
            for detected_box in tl_boxes:
                #rospy.loginfo("[TL Detector] Detected box: " +str(detected_box))
                cropped_tls.append(
                    cv2.resize(
                        image[detected_box[0] +10 : int(detected_box[0] + (detected_box[2] - detected_box[0])/2.5), detected_box[1] + 5: detected_box[3] - 5],
                        (32, 32)))

            # TODO(saajan): Remove this testing code
            # for cropped_tl in cropped_tls:
            #     try:
            #         self.cropped_boxes.publish(self.bridge.cv2_to_imgmsg(cropped_tl, "rgb8"))
            #     except CvBridgeError as e:
            #         print(e)



            return cropped_tls


    def normalized_box_to_pixels(self, box, image):
        height, width = image.shape[0], image.shape[1]

        return [int(height * box[0]), int(width * box[1]),
                int(height * box[2]), int(width * box[3])]
