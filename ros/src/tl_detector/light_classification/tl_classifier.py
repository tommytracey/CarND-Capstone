from styx_msgs.msg import TrafficLight
import rospy

import tensorflow as tf
import cv2
import numpy as np
from collections import Counter
import glob

class TLClassifier(object):
    def __init__(self, model_path, confidence_threshold):

        self.confidence_threshold = confidence_threshold

        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.graph.as_default():
            graph_def = tf.GraphDef()

            with tf.gfile.GFile(model_path, 'rb') as graph_file:
                read_graph_file = graph_file.read()

                graph_def.ParseFromString(read_graph_file)
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.graph, config=config)

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.num_detections =self.graph.get_tensor_by_name('num_detections:0')
            # For each detection, it's corresponding bounding box, class and score:
            # self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.scores =self.graph.get_tensor_by_name('detection_scores:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self.graph.as_default():
            cv2_image_expanded = np.expand_dims(image, axis=0)

            (classes, scores) = self.sess.run(
                [self.classes, self.scores],
                feed_dict={self.image_tensor: cv2_image_expanded})
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            rospy.loginfo("[TL Detector] Results:")
            rospy.loginfo("[TL Detector] Classes: {0}".format(classes))
            rospy.loginfo("[TL Detector] Scores: {0}".format(scores))


#             ##############
#             ## Strategy 1: pick the most frequent prediction (consider replacing
#             # with weighted frequency)
#             #
#             #
#             class_predictions_above_threshold = []
#             for detected_class, score in zip(classes, scores):
#                 if score > self.confidence_threshold:
#                     class_predictions_above_threshold.append(detected_class)

#             if (len(class_predictions_above_threshold) == 0):
#                 rospy.loginfo("[Detector] Predicted Class: Unknown")
#                 return TrafficLight.UNKNOWN

#             class_predictions = Counter(class_predictions_above_threshold)
#             most_frequent_class = class_predictions.most_common(1)[0][0]

#             if most_frequent_class == 1:
#                 rospy.loginfo("[Detector] Predicted Class: Red")
#                 return TrafficLight.RED
#             elif most_frequent_class == 2:
#                 rospy.loginfo("[Detector] Predicted Class: Yellow")
#                 return TrafficLight.YELLOW
#             elif most_frequent_class == 3:
#                 rospy.loginfo("[Detector] Predicted Class: Green")
#                 return TrafficLight.GREEN
#             #
#             #
#             ##############


            ##############
            ## Strategy 2: pick the first prediction above threshold
            for detected_class, score in zip(classes, scores):
                if score > self.confidence_threshold:
                    if detected_class == 1:
                        rospy.loginfo("[Detector] Predicted Class: Red")
                        return TrafficLight.RED
                    elif detected_class == 2:
                        rospy.loginfo("[Detector] Predicted Class: Yellow")
                        return TrafficLight.YELLOW
                    elif detected_class == 3:
                        rospy.loginfo("[Detector] Predicted Class: Green")
                        return TrafficLight.GREEN

            #
            #
            ##############


        rospy.loginfo("[TL Detector] Predicted Class: Unknown")
        return TrafficLight.UNKNOWN
