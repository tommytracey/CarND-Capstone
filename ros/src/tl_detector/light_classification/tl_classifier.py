from styx_msgs.msg import TrafficLight
import rospy


import tensorflow as tf
import cv2
import numpy as np
from collections import Counter
import glob

class TLClassifier(object):
    def __init__(self, model_path, confidence_threshold):

        self.graph = tf.graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = true

        with self.graph.as_default():
            graph_def = tf.graph_def

            with tf.gfile.GFile(model_path, 'rb') as graph_file:
                read_graph_file = graph_file.read()

                graph_def.ParseFromString(read_graph_file)
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')
            # For each detection, it's corresponding bounding box, class and score:
            # self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')


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

            class_predictions_above_threshold = []
            for detected_class, score in zip(classes, scores):
                if score > self.confidence_threshold:
                    class_predictions_above_threshold.append(detected_class)

            class_predictions = Counter(class_predictions_above_threshold)
            most_frequent_class = class_predictions.most_common(1)[0][0]

            if most_frequent_class == 1:
                rospy.loginfo("[Detector] Predicted Class: Red")
                return TrafficLight.RED
            elif most_frequent_class == 2:
                rospy.loginfo("[Detector] Predicted Class: Yellow")
                return TrafficLight.YELLOW
            elif most_frequent_class == 3:
                rospy.loginfo("[Detector] Predicted Class: Green")
                return TrafficLight.GREEN

        rospy.loginfo("[Detector] Predicted Class: Unknown")
        return TrafficLight.UNKNOWN

if __name__ == '__main__':
    classifier =TLClassifier(modelpath= "../data/saved_models/model_real/frozen_inference_graph.pb", threshold=0.3)

    images_path = "../data/sample_images/*.png"
    for image_path, expected_label in zip(
            sorted(glob.glob(images_path)),
            [3, 3, 3, 1, 1, 1, 2, 2, 2]):

        image = cv2.imread(image_path)
        traffic_light = classifier.get_classification(image)

        if traffic_light != expected_label:
            raise ValueError("Wrong class for image {0}".format(image_path))
