import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
devices = tf.config.experimental.list_physical_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as util
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('framework_type', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('model_weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('img_size', 416, 'resize images to')
flags.DEFINE_boolean('is_tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('yolo_model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('input_video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output_video', None, 'path to output video')
flags.DEFINE_string('video_codec', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou_threshold', 0.45, 'iou threshold')
flags.DEFINE_float('confidence_score', 0.50, 'score threshold')
flags.DEFINE_boolean('display_output', False, 'dont show video output')
flags.DEFINE_boolean('detailed_info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count_objects', False, 'count objects being tracked on screen')

def main(_args):
    max_distance = 0.4
    budget = None
    max_overlap = 1.0
    
    model_file = 'model_data/mars-small128.pb'
    encoder_model = gdet.create_box_encoder(model_file, batch_size=1)
    distance_metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_distance, budget)
    tracker_instance = Tracker(distance_metric)

    config_options = ConfigProto()
    config_options.gpu_options.allow_growth = True
    session_instance = InteractiveSession(config=config_options)
    strides, anchors, num_classes, xy_scale = util.load_config(FLAGS)
    img_size = FLAGS.img_size
    video_file = FLAGS.input_video

    if FLAGS.framework_type == 'tflite':
        tflite_interpreter = tf.lite.Interpreter(model_path=FLAGS.model_weights)
        tflite_interpreter.allocate_tensors()
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        loaded_model = tf.saved_model.load(FLAGS.model_weights, tags=[tag_constants.SERVING])
        inference = loaded_model.signatures['serving_default']

    try:
        video_capture = cv2.VideoCapture(int(video_file))
    except:
        video_capture = cv2.VideoCapture(video_file)

    output_writer = None

    if FLAGS.output_video:
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.video_codec)
        output_writer = cv2.VideoWriter(FLAGS.output_video, codec, fps, (width, height))

    frame_count = 0
    total_counted = 0
    while True:
        ret_val, frame_data = video_capture.read()
        if ret_val:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_data)
        else:
            print('Video ended or failed, try a different format!')
            break
        frame_count += 1
        print('Frame #: ', frame_count)
        frame_dims = frame_data.shape[:2]
        img_data = cv2.resize(frame_data, (img_size, img_size))
        img_data = img_data / 255.
        img_data = img_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework_type == 'tflite':
            tflite_interpreter.set_tensor(input_details[0]['index'], img_data)
            tflite_interpreter.invoke()
            predictions = [tflite_interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.yolo_model == 'yolov3' and FLAGS.is_tiny:
                boxes, prediction_confidence = filter_boxes(predictions[1], predictions[0], score_threshold=0.25,
                                                             input_shape=tf.constant([img_size, img_size]))
            else:
                boxes, prediction_confidence = filter_boxes(predictions[0], predictions[1], score_threshold=0.25,
                                                             input_shape=tf.constant([img_size, img_size]))
        else:
            batch_input = tf.constant(img_data)
            bbox_predictions = inference(batch_input)
            for key, value in bbox_predictions.items():
                boxes = value[:, :, 0:4]
                prediction_confidence = value[:, :, 4:]

        boxes, scores, class_indices, valid_det = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                prediction_confidence, (tf.shape(prediction_confidence)[0], -1, tf.shape(prediction_confidence)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou_threshold,
            score_threshold=FLAGS.confidence_score
        )

        num_items = valid_det.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_items)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_items)]
        classes = class_indices.numpy()[0]
        classes = classes[0:int(num_items)]

        orig_height, orig_width, _ = frame_data.shape
        bboxes = util.format_boxes(bboxes, orig_height, orig_width)

        predictions = [bboxes, scores, classes, num_items]

        class_labels = util.read_class_names(cfg.YOLO.CLASSES)
        
        allowed_labels = ['bean']

        names_list = []
        removed_indices = []
        for i in range(num_items):
            class_index = int(classes[i])
            class_name = class_labels[class_index]
            if class_name not in allowed_labels:
                removed_indices.append(i)
            else:
                names_list.append(class_name)
        names_list = np.array(names_list)
        count_of_objects = len(names_list)
        if FLAGS.count_objects:
            cv2.putText(frame_data, "Tracked Objects: {}".format(count_of_objects), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Tracked Objects: {}".format(count_of_objects))
        bboxes = np.delete(bboxes, removed_indices, axis=0)
        scores = np.delete(scores, removed_indices, axis=0)

        feature_data = encoder_model(frame_data, bboxes)
        detected_items = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names_list, feature_data)]

        color_map = plt.get_cmap('tab20b')
        color_palette = [color_map(i)[:3] for i in np.linspace(0, 1, 20)]

        boxes_array = np.array([d.tlwh for d in detected_items])
        scores_array = np.array([d.confidence for d in detected_items])
        classes_array = np.array([d.class_name for d in detected_items])
        indices_array = preprocessing.non_max_suppression(boxes_array, classes_array, max_overlap, scores_array)
        detected_items = [detected_items[i] for i in indices_array]       

        tracker_instance.predict()
        tracker_instance.update(detected_items)

        index = 0
        
        for track in tracker_instance.tracks:
            index += 1
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            box_coords = track.to_tlbr()
            class_label = track.get_class()
            
            # Correctly count tracked objects
            if (track.track_id >= total_counted) and (frame_count % 10 == 0):
                total_counted = track.track_id

            save_path = "/Users/user/Desktop/outputs"
            xmin, ymin, xmax, ymax = int(box_coords[0]), int(box_coords[1]), int(box_coords[2]), int(box_coords[3])
            cropped_img = frame_data[int(ymin)-15:int(ymax)+15, int(xmin)-15:int(xmax)+15, ::-1]
            image_name = "ID" + '_' + str(track.track_id) + '.jpg'
            image_full_path = os.path.join(save_path, image_name)
            cv2.imwrite(image_full_path, cropped_img)

            color = color_palette[track.track_id % len(color_palette)]
            color = [int(c * 255) for c in color]

            label_text = f'ID: {track.track_id} {class_label} {track.confidence:.2f}'
            frame_data = util.draw_boxes(frame_data, box_coords, color, label_text)

        if FLAGS.output_video:
            output_writer.write(frame_data)

        if FLAGS.display_output:
            cv2.imshow("Detection", frame_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if output_writer:
        output_writer.release()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
