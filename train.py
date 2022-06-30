import datetime
import contextlib
import tensorflow as tf
import tensorflow_addons as tfa
# import tensorflow_model_optimization as tfmot
# tf.config.experimental_run_functions_eagerly(True)
# tf.debugging.enable_check_numerics()

# it s recommanded to use absl for tf 2.0
from absl import app
from absl import flags
from absl import logging
import os
from model import MaskED
from data import dataset_coco
from loss import loss
from utils import learning_rate_schedule
from utils import coco_evaluation
from utils import standard_fields
from config import Config

import numpy as np
import cv2
from google.protobuf import text_format
from protos import string_int_label_map_pb2
import time
from tqdm import tqdm

tf.random.set_seed(123)
physical_devices = tf.config.list_physical_devices('GPU')
try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
        print("Invalid device or cannot modify virtual devices once initialized.")
        pass

FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_train_dir', './data/coco/train',
                    'directory of training tfrecord')
flags.DEFINE_string('tfrecord_val_dir', './data/coco/val',
                    'directory of validation tfrecord')
flags.DEFINE_string('checkpoints_dir', './checkpoints',
                    'directory for saving checkpoints')
flags.DEFINE_string('pretrained_checkpoints', '',
                    'path to pretrained checkpoints')
flags.DEFINE_string('logs_dir', './logs',
                    'directory for saving logs')
flags.DEFINE_string('saved_models_dir', './saved_models',
                    'directory for exporting saved_models')
flags.DEFINE_string('label_map', './label_map.pbtxt',
                    'path to label_map.pbtxt')
flags.DEFINE_float('print_interval', 100,
                   'number of iteration between printing loss')
flags.DEFINE_float('save_interval', 10000,
                   'number of iteration between saving model(checkpoint)')
flags.DEFINE_float('valid_iter', 20,
                   'number of iteration during validation')

'''
def _get_categories_list():
  
'''
def _validate_label_map(label_map):
  # https://github.com/tensorflow/models/blob/
  # 67fd2bef6500c14b95e0b0de846960ed13802405/research/object_detection/utils/
  # label_map_util.py#L34
  """Checks if a label map is valid.
  Args:
    label_map: StringIntLabelMap to validate.
  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 0:
      raise ValueError('Label map ids should be >= 0.')
    if (item.id == 0 and item.name != 'background' and
        item.display_name != 'background'):
      raise ValueError('Label map id 0 is reserved for the background label')

def load_labelmap(path):
  # https://github.com/tensorflow/models/blob/
  # 67fd2bef6500c14b95e0b0de846960ed13802405/research/object_detection/utils/
  # label_map_util.py#L159
  """Loads label map proto.
  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with tf.io.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map

def _get_categories_list(label_map_path):
    # https://github.com/tensorflow/models/blob/\
    # 67fd2bef6500c14b95e0b0de846960ed13802405/research/cognitive_planning/
    # label_map_util.py#L73
    '''
    return [{
          'id': 1,
          'name': 'person'
      }, {
          'id': 2,
          'name': 'dog'
      }, {
          'id': 3,
          'name': 'cat'
      }]
    '''
    label_map = load_labelmap(label_map_path)
    categories = []
    list_of_ids_already_added = []
    for item in label_map.item:
        name = item.name
        if item.id not in list_of_ids_already_added:
          list_of_ids_already_added.append(item.id)
          categories.append({'id': item.id, 'name': name})
    return categories

def main(argv):
    # set up Grappler for graph optimization
    # Ref: https://www.tensorflow.org/guide/graph_optimization
    @contextlib.contextmanager
    def options(options):
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(options)
        try:
            yield
        finally:
            tf.config.optimizer.set_experimental_options(old_opts)


    mirrored_strategy = tf.distribute.MirroredStrategy()    

    with mirrored_strategy.scope():
        config = Config()
        config.display()
        model = MaskED(config)

        # -----------------------------------------------------------------
        # Choose the Optimizor, Loss Function, and Metrics, learning rate schedule 

        # add weight decay
        # Skip gamma and beta weights of batch normalization layers.
        def add_weight_decay(model, weight_decay):
            # https://github.com/keras-team/keras/issues/12053
            if (weight_decay is None) or (weight_decay == 0.0):
                return

            # recursion inside the model
            def add_decay_loss(m, factor):
                if isinstance(m, tf.keras.Model):
                    for layer in m.layers:
                        add_decay_loss(layer, factor)
                else:
                    for param in m.trainable_weights:
                      if 'gamma' not in param.name and 'beta' not in param.name:
                        with tf.keras.backend.name_scope('weight_regularizer'):
                            regularizer = lambda: tf.keras.regularizers.l2(factor)(param)
                            m.add_loss(regularizer)

            # weight decay and l2 regularization differs by a factor of 2
            # because the weights are updated as w := w - l_r * L(w,x) - 2 * l_r * l2 * w
            # where L-r is learning rate, l2 is L2 regularization factor. The whole (2 * l2)
            # forms a weight decay factor. So, in pytorch where weight decay is directly given
            # and in tf where l2 regularization has to be used differs by a factor of 2.
            add_decay_loss(model, weight_decay/2.0)
            return

        add_weight_decay(model, config.WEIGHT_DECAY)   

        logging.info("Initiate the Optimizer and Loss function...")
        if config.OPTIMIZER == 'SGD':
          logging.info("Using SGD optimizer")
          lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
             [config.N_WARMUP_STEPS, int(0.35*config.TRAIN_ITER), int(0.75*config.TRAIN_ITER), int(0.875*config.TRAIN_ITER), int(0.9375*config.TRAIN_ITER)], 
             [config.WARMUP_LR, config.LEARNING_RATE, 0.1*config.LEARNING_RATE, 0.01*config.LEARNING_RATE, 0.001*config.LEARNING_RATE, 0.0001*config.LEARNING_RATE])
          if config.GRADIENT_CLIP_NORM is not None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config.LEARNING_MOMENTUM, clipnorm=config.GRADIENT_CLIP_NORM)
          else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config.LEARNING_MOMENTUM)
        elif config.OPTIMIZER == 'Adam':
          logging.info("Using Adam optimizer")
          lr_schedule = learning_rate_schedule.LearningRateSchedule(
            warmup_steps=config.N_WARMUP_STEPS, 
            warmup_lr=config.WARMUP_LR,
            initial_lr=config.LEARNING_RATE, 
            total_steps=config.LR_TOTAL_STEPS)
          if config.GRADIENT_CLIP_NORM is not None:
            optimizer = tf.keras.optimizers.Adam(
              learning_rate=lr_schedule, clipnorm=10)
          else:
            optimizer = tf.keras.optimizers.Adam(
              learning_rate=lr_schedule)

        # setup checkpoints manager
        checkpoint = tf.train.Checkpoint(
          step=tf.Variable(1), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=FLAGS.checkpoints_dir, max_to_keep=5
        )
        # restore from latest checkpoint and iteration
        status = checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            logging.info("Restored from {}".format(manager.latest_checkpoint))
        else:
            if FLAGS.pretrained_checkpoints != '':
              feature_extractor_model = tf.train.Checkpoint(
                backbone=model.backbone)
              ckpt = tf.train.Checkpoint(model=feature_extractor_model)
              ckpt.restore(FLAGS.pretrained_checkpoints).\
                expect_partial().assert_existing_objects_matched()
              logging.info("Backbone restored from {}".format(
                FLAGS.pretrained_checkpoints))
            else:
              logging.info("Initializing from scratch.")

    BATCH_SIZE_PER_REPLICA = config.BATCH_SIZE
    global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     mirrored_strategy.num_replicas_in_sync)
    
    # -----------------------------------------------------------------
    # Creating dataloaders for training and validation
    logging.info("Creating the training dataloader from: %s..." % \
      FLAGS.tfrecord_train_dir)
    train_dataset = dataset_coco.prepare_dataloader(
      config, 
      tfrecord_dir=FLAGS.tfrecord_train_dir,
      feature_map_size=model.feature_map_size,
      batch_size=global_batch_size,
      subset='train')
    train_dataset_dist = mirrored_strategy.experimental_distribute_dataset(train_dataset)

    logging.info("Creating the validation dataloader from: %s..." % \
      FLAGS.tfrecord_val_dir)
    valid_dataset = dataset_coco.prepare_dataloader(
      config,
      tfrecord_dir=FLAGS.tfrecord_val_dir,
      feature_map_size=model.feature_map_size,
      batch_size=config.BATCH_SIZE,
      subset='val')

    criterion = loss.Loss(config)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
    loc = tf.keras.metrics.Mean('loc_loss', dtype=tf.float32)
    conf = tf.keras.metrics.Mean('conf_loss', dtype=tf.float32)
    mask = tf.keras.metrics.Mean('mask_loss', dtype=tf.float32)
    mask_iou = tf.keras.metrics.Mean('mask_iou_loss', dtype=tf.float32)
    v_loc = tf.keras.metrics.Mean('vloc_loss', dtype=tf.float32)
    v_conf = tf.keras.metrics.Mean('vconf_loss', dtype=tf.float32)
    v_mask = tf.keras.metrics.Mean('vmask_loss', dtype=tf.float32)
    v_mask_iou = tf.keras.metrics.Mean('vmask_iou_loss', dtype=tf.float32)
    global_norm = tf.keras.metrics.Mean('global_norm', dtype=tf.float32)
    precision_mAP = tf.keras.metrics.Mean('precision_mAP', dtype=tf.float32)
    precision_mAP_50IOU = tf.keras.metrics.Mean('precision_mAP_50IOU', 
      dtype=tf.float32)
    precision_mAP_75IOU = tf.keras.metrics.Mean('precision_mAP_75IOU', 
      dtype=tf.float32)
    precision_mAP_small = tf.keras.metrics.Mean('precision_mAP_small', 
      dtype=tf.float32)
    precision_mAP_medium = tf.keras.metrics.Mean('precision_mAP_medium', 
      dtype=tf.float32)
    precision_mAP_large = tf.keras.metrics.Mean('precision_mAP_large', 
      dtype=tf.float32)
    recall_AR_1 = tf.keras.metrics.Mean('recall_AR_1', 
      dtype=tf.float32)
    recall_AR_10 = tf.keras.metrics.Mean('recall_AR_10', 
      dtype=tf.float32)
    recall_AR_100 = tf.keras.metrics.Mean('recall_AR_100', 
      dtype=tf.float32)
    recall_AR_100_small = tf.keras.metrics.Mean('recall_AR_100_small', 
      dtype=tf.float32)
    recall_AR_100_medium = tf.keras.metrics.Mean('recall_AR_100_medium', 
      dtype=tf.float32)
    recall_AR_100_large = tf.keras.metrics.Mean('recall_AR_100_large', 
      dtype=tf.float32)

    # -----------------------------------------------------------------

    # Setup the TensorBoard for better visualization
    # Ref: https://www.tensorflow.org/tensorboard/get_started
    logging.info("Setup the TensorBoard...")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(FLAGS.logs_dir,'train')
    test_log_dir = os.path.join(FLAGS.logs_dir, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # -----------------------------------------------------------------
    # Start the Training and Validation Process
    logging.info("Start the training process...")

    # COCO evalator for showing MAP
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(
      _get_categories_list(FLAGS.label_map))

    best_val = 1e10
    iterations = checkpoint.step.numpy()

    def train_step(image, labels):
        with options({'constant_folding': True,
                      'layout_optimize': True,
                      'loop_optimization': True,
                      'arithmetic_optimization': True,
                      'remapping': True}):
            with tf.GradientTape() as tape:
                output = model(image, training=True)

                loc_loss, conf_loss, mask_loss, mask_iou_loss, \
                    = criterion(model, output, labels, config.NUM_CLASSES+1, image)

                loc_loss = tf.nn.compute_average_loss(loc_loss, global_batch_size=global_batch_size)
                conf_loss = tf.nn.compute_average_loss(conf_loss, global_batch_size=global_batch_size)
                mask_loss = tf.nn.compute_average_loss(mask_loss, global_batch_size=global_batch_size)
                mask_iou_loss = tf.nn.compute_average_loss(mask_iou_loss, global_batch_size=global_batch_size)
    
                total_loss = loc_loss + conf_loss + mask_loss + mask_iou_loss
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return (loc_loss, conf_loss, mask_loss, mask_iou_loss, total_loss, grads)

    @tf.function
    def distributed_train_step(image, labels):
        per_replica_losses = mirrored_strategy.run(train_step, args=(image, labels,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

    for image, labels in train_dataset_dist:
        # check iteration and change the learning rate
        if iterations > config.TRAIN_ITER:
            break

        checkpoint.step.assign_add(1)
        iterations += 1

        loc_loss, conf_loss, mask_loss, mask_iou_loss, total_loss, grads = distributed_train_step(image, labels)
        global_norm.update_state(tf.linalg.global_norm(grads))
        train_loss.update_state(total_loss)
        loc.update_state(loc_loss)
        conf.update_state(conf_loss)
        mask.update_state(mask_loss)
        mask_iou.update_state(mask_iou_loss)

        with train_summary_writer.as_default():
          with tf.name_scope("loss_train"):
            tf.summary.scalar('Total loss', 
              train_loss.result(), step=iterations)

            tf.summary.scalar('Loc loss', 
              loc.result(), step=iterations)

            tf.summary.scalar('Conf loss', 
              conf.result(), step=iterations)

            tf.summary.scalar('Mask loss', 
              mask.result(), step=iterations)

            tf.summary.scalar('Mask IOU loss', 
              mask_iou.result(), step=iterations)

          with tf.name_scope("norm"):
            tf.summary.scalar('Global Norm', 
              global_norm.result(), step=iterations)

        if iterations and iterations % FLAGS.print_interval == 0:
            logging.info(
                ("Iteration {}, LR: {}, Total Loss: {:.4f}, B: {:.4f},  "
                  "C: {:.4f}, M: {:.4f}, I: {:.4f}, "
                  "global_norm:{:.4f} ").format(
                iterations,
                optimizer._decayed_lr(var_dtype=tf.float32),
                train_loss.result(), 
                loc.result(),
                conf.result(),
                mask.result(),
                mask_iou.result(),
                global_norm.result()
            ))

        if iterations and iterations % FLAGS.save_interval == 0:
            # save checkpoint
            save_path = manager.save()

            logging.info("Saved checkpoint for step {}: {}".format(
              int(iterations), save_path))

            # validation
            valid_iter = 0
            for valid_image, valid_labels in tqdm(valid_dataset):
                if valid_iter > FLAGS.valid_iter:
                    break
                # calculate validation loss
                with options({'constant_folding': True,
                              'layout_optimize': True,
                              'loop_optimization': True,
                              'arithmetic_optimization': True,
                              'remapping': True}):
                    output = model(valid_image, training=False)

                    valid_loc_loss, valid_conf_loss, valid_mask_loss, \
                    valid_mask_iou_loss = \
                    criterion(model, output, valid_labels, config.NUM_CLASSES+1)

                    valid_total_loss = sum(valid_loc_loss + valid_conf_loss + valid_mask_loss + valid_mask_iou_loss)
                    valid_loss.update_state(valid_total_loss)

                    _h = valid_image.shape[1]
                    _w = valid_image.shape[2]
                    
                    for b in range(config.BATCH_SIZE):
                        image_id = int(time.time()*1000000)
                        gt_num_box = valid_labels['num_obj'][b].numpy()
                        gt_boxes = valid_labels['boxes_norm'][b][:gt_num_box]
                        gt_boxes = gt_boxes.numpy()*np.array([_h,_w,_h,_w])
                        gt_classes = valid_labels['classes'][b][:gt_num_box].numpy()
                        gt_masks = valid_labels['mask_target'][b][:gt_num_box].numpy()

                        gt_masked_image = np.zeros((gt_num_box, _h, _w), dtype=np.uint8)
                        for _b in range(gt_num_box):
                            box = gt_boxes[_b]
                            box = np.round(box).astype(int)
                            (startY, startX, endY, endX) = box.astype("int")
                            boxW = endX - startX
                            boxH = endY - startY
                            if boxW > 0 and boxH > 0:
                              _m = cv2.resize(gt_masks[_b].astype("uint8"), (boxW, boxH))
                              gt_masked_image[_b][startY:endY, startX:endX] = _m

                        coco_evaluator.add_single_ground_truth_image_info(
                            image_id='image'+str(image_id),
                            groundtruth_dict={
                              standard_fields.InputDataFields.groundtruth_boxes: gt_boxes,
                              standard_fields.InputDataFields.groundtruth_classes: gt_classes,
                              standard_fields.InputDataFields.groundtruth_instance_masks: gt_masked_image
                            })

                        det_num = np.count_nonzero(output['detection_scores'][0].numpy()> 0.05)

                        det_boxes = output['detection_boxes'][b][:det_num]
                        det_boxes = det_boxes.numpy()*np.array([_h,_w,_h,_w])
                        det_masks = output['detection_masks'][b][:det_num].numpy()
                        det_masks = (det_masks > 0.5).astype("uint8")

                        det_scores = output['detection_scores'][b][:det_num].numpy()
                        det_classes = output['detection_classes'][b][:det_num].numpy().astype(int)

                        det_masked_image = np.zeros((det_num, _h, _w), dtype=np.uint8)
                        for _b in range(det_num):
                            box = det_boxes[_b]
                            _c = det_classes[_b] - 1
                            _m = det_masks[_b][:, :, _c]
                            box = np.round(box).astype(int)
                            (startY, startX, endY, endX) = box.astype("int")
                            boxW = endX - startX
                            boxH = endY - startY
                            if boxW > 0 and boxH > 0:
                              _m = cv2.resize(_m, (boxW, boxH))
                              det_masked_image[_b][startY:endY, startX:endX] = _m
                        
                        coco_evaluator.add_single_detected_image_info(
                            image_id='image'+str(image_id),
                            detections_dict={
                                standard_fields.DetectionResultFields.detection_boxes: det_boxes,
                                standard_fields.DetectionResultFields.detection_scores: det_scores,
                                standard_fields.DetectionResultFields.detection_classes: det_classes,
                                standard_fields.DetectionResultFields.detection_masks: det_masked_image
                            })

                v_loc.update_state(valid_loc_loss)
                v_conf.update_state(valid_conf_loss)
                v_mask.update_state(valid_mask_loss)
                v_mask_iou.update_state(valid_mask_iou_loss)
                valid_iter += 1

            metrics = coco_evaluator.evaluate()
            precision_mAP.update_state(
              metrics['DetectionMasks_Precision/mAP'])
            precision_mAP_50IOU.update_state(
              metrics['DetectionMasks_Precision/mAP@.50IOU'])
            precision_mAP_75IOU.update_state(
              metrics['DetectionMasks_Precision/mAP@.75IOU'])
            precision_mAP_small.update_state(
              metrics['DetectionMasks_Precision/mAP (small)'])
            precision_mAP_medium.update_state(
              metrics['DetectionMasks_Precision/mAP (medium)'])
            precision_mAP_large.update_state(
              metrics['DetectionMasks_Precision/mAP (large)'])
            recall_AR_1.update_state(
              metrics['DetectionMasks_Recall/AR@1'])
            recall_AR_10.update_state(
              metrics['DetectionMasks_Recall/AR@10'])
            recall_AR_100.update_state(
              metrics['DetectionMasks_Recall/AR@100'])
            recall_AR_100_small.update_state(
              metrics['DetectionMasks_Recall/AR@100 (small)'])
            recall_AR_100_medium.update_state(
              metrics['DetectionMasks_Recall/AR@100 (medium)'])
            recall_AR_100_large.update_state(
              metrics['DetectionMasks_Recall/AR@100 (large)'])

            coco_evaluator.clear()

            with test_summary_writer.as_default():
              with tf.name_scope("loss_val"):
                tf.summary.scalar('V Total loss', 
                  valid_loss.result(), step=iterations)

                tf.summary.scalar('V Loc loss', 
                  v_loc.result(), step=iterations)

                tf.summary.scalar('V Conf loss', 
                  v_conf.result(), step=iterations)

                tf.summary.scalar('V Mask loss', 
                  v_mask.result(), step=iterations)

                tf.summary.scalar('V Mask IOU loss', 
                  v_mask_iou.result(), step=iterations)

              with tf.name_scope("precision"):
                tf.summary.scalar('Precision mAP', 
                  precision_mAP.result(), step=iterations)

                tf.summary.scalar('Precision mAP@.50IOU', 
                  precision_mAP_50IOU.result(), step=iterations)

                tf.summary.scalar('precision mAP@.75IOU', 
                  precision_mAP_75IOU.result(), step=iterations)

                tf.summary.scalar('precision mAP (small)', 
                  precision_mAP_small.result(), step=iterations)

                tf.summary.scalar('precision mAP (medium)', 
                  precision_mAP_medium.result(), step=iterations)

                tf.summary.scalar('precision mAP (large)', 
                  precision_mAP_large.result(), step=iterations)

              with tf.name_scope("recall"):
                tf.summary.scalar('recall AR@1', 
                  recall_AR_1.result(), step=iterations)

                tf.summary.scalar('recall AR@10', 
                  recall_AR_10.result(), step=iterations)

                tf.summary.scalar('recall AR@100', 
                  recall_AR_100.result(), step=iterations)

                tf.summary.scalar('recall AR@100 (small)', 
                  recall_AR_100_small.result(), step=iterations)

                tf.summary.scalar('recall AR@100 (medium)', 
                  recall_AR_100_medium.result(), step=iterations)

                tf.summary.scalar('recall AR@100 (large)', 
                  recall_AR_100_large.result(), step=iterations)

            train_template = ("Iteration {}, Train Loss: {}, Loc Loss: {},  "
              "Conf Loss: {}, Mask Loss: {}, Mask IOU Loss: {}, "
              "Global Norm: {}")

            valid_template = ("Iteration {}, Precision mAP: {}, Precision "
              "mAP@.50IOU: {}, precision mAP@.75IOU: {}, precision mAP (small)"
              ": {}, precision mAP (medium): {}, precision mAP (large): {}, "
              " recall AR@1: {}, recall AR@10: {}, recall AR@100: {}, recall "
              "AR@100 (small): {}, recall AR@100 (medium): {}, recall AR@100 "
              "(large): {}\nValid Loss: {}, V Loc Loss: {},  "
              "V Conf Loss: {}, V Mask Loss: {}, V Mask IOU Loss: {}")

            logging.info(train_template.format(iterations + 1,
                                        train_loss.result(),
                                        loc.result(),
                                        conf.result(),
                                        mask.result(),
                                        mask_iou.result(),
                                        global_norm.result()))
            logging.info(valid_template.format(iterations + 1,
                                        precision_mAP.result(),
                                        precision_mAP_50IOU.result(),
                                        precision_mAP_75IOU.result(),
                                        precision_mAP_small.result(),
                                        precision_mAP_medium.result(),
                                        precision_mAP_large.result(),
                                        recall_AR_1.result(),
                                        recall_AR_10.result(),
                                        recall_AR_100.result(),
                                        recall_AR_100_small.result(),
                                        recall_AR_100_medium.result(),
                                        recall_AR_100_large.result(),
                                        valid_loss.result(),
                                        v_loc.result(),
                                        v_conf.result(),
                                        v_mask.result(),
                                        v_mask_iou.result()))
            if valid_loss.result() < best_val:
                best_val = valid_loss.result()
                save_options = tf.saved_model.SaveOptions(
                  namespace_whitelist=['Addons'])

                model.save(os.path.join(
                  FLAGS.saved_models_dir, 
                  'saved_model_'+ str(valid_loss.result().numpy())), 
                options=save_options)

            # reset the metrics
            train_loss.reset_states()
            loc.reset_states()
            conf.reset_states()
            mask.reset_states()
            mask_iou.reset_states()
            global_norm.reset_states()

            valid_loss.reset_states()
            v_loc.reset_states()
            v_conf.reset_states()
            v_mask.reset_states()
            v_mask_iou.reset_states()
            precision_mAP.reset_states()
            precision_mAP_50IOU.reset_states()
            precision_mAP_75IOU.reset_states()
            precision_mAP_small.reset_states()
            precision_mAP_medium.reset_states()
            precision_mAP_large.reset_states()
            recall_AR_1.reset_states()
            recall_AR_10.reset_states()
            recall_AR_100.reset_states()
            recall_AR_100_small.reset_states()
            recall_AR_100_medium.reset_states()
            recall_AR_100_large.reset_states()


if __name__ == '__main__':
    app.run(main)
