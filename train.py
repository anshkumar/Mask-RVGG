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
flags.DEFINE_bool('multi_gpu', False,
                   'whether to use multi gpu training or not.')

# def _get_categories_list():
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

def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5

def unitwise_norm(x):
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2,]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor=0.01,
                       eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads

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
              # if 'gamma' not in param.name and 'beta' not in param.name:
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

def get_optimizer(config):
    logging.info("Initiate the Optimizer and Loss function...")
    if config.OPTIMIZER == 'SGD':
      logging.info("Using SGD optimizer")
      #lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
      #   [config.N_WARMUP_STEPS, int(0.35*config.TRAIN_ITER), int(0.75*config.TRAIN_ITER), int(0.875*config.TRAIN_ITER), int(0.9375*config.TRAIN_ITER)], 
      #   [config.WARMUP_LR, config.LEARNING_RATE, 0.1*config.LEARNING_RATE, 0.01*config.LEARNING_RATE, 0.001*config.LEARNING_RATE, 0.0001*config.LEARNING_RATE])
      lr_schedule = learning_rate_schedule.LearningRateSchedule(
        warmup_steps=config.N_WARMUP_STEPS, 
        warmup_lr=config.WARMUP_LR,
        initial_lr=config.LEARNING_RATE, 
        total_steps=config.LR_TOTAL_STEPS)
      if config.GRADIENT_CLIP_NORM is not None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config.LEARNING_MOMENTUM, clipnorm=config.GRADIENT_CLIP_NORM, nesterov=True)
      else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config.LEARNING_MOMENTUM)
    elif config.OPTIMIZER == 'Adam':
      logging.info("Using Adam optimizer")
      lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
          [config.N_WARMUP_STEPS, int(0.35*config.TRAIN_ITER), int(0.75*config.TRAIN_ITER), int(0.875*config.TRAIN_ITER), int(0.9375*config.TRAIN_ITER)], 
          [config.WARMUP_LR, config.LEARNING_RATE, 0.1*config.LEARNING_RATE, 0.01*config.LEARNING_RATE, 0.001*config.LEARNING_RATE, 0.0001*config.LEARNING_RATE])
      # lr_schedule = learning_rate_schedule.LearningRateSchedule(
      #   warmup_steps=config.N_WARMUP_STEPS, 
      #   warmup_lr=config.WARMUP_LR,
      #   initial_lr=config.LEARNING_RATE, 
      #   total_steps=config.LR_TOTAL_STEPS)
      if config.GRADIENT_CLIP_NORM is not None:
        optimizer = tf.keras.optimizers.Adam(
          learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
      else:
        optimizer = tf.keras.optimizers.Adam(
          learning_rate=lr_schedule)
    elif config.OPTIMIZER == 'AdamW':
      lr_schedule = learning_rate_schedule.LearningRateSchedule(
        warmup_steps=config.N_WARMUP_STEPS, 
        warmup_lr=config.WARMUP_LR,
        initial_lr=config.LEARNING_RATE, 
        total_steps=config.LR_TOTAL_STEPS)
      if config.GRADIENT_CLIP_NORM is not None:
        optimizer = tfa.optimizers.AdamW(
          learning_rate=lr_schedule, 
          weight_decay=config.WEIGHT_DECAY, clipnorm=config.GRADIENT_CLIP_NORM)
      else:
        optimizer = tfa.optimizers.AdamW(
          learning_rate=lr_schedule,
          weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'AdaBelief':
      optimizer = tfa.optimizers.AdaBelief(
          lr=config.LEARNING_RATE,
          total_steps=config.LR_TOTAL_STEPS,
          warmup_proportion=config.N_WARMUP_STEPS/config.LR_TOTAL_STEPS,
          min_lr=config.LEARNING_RATE*config.LEARNING_RATE,
          rectify=True)

    return optimizer

def get_checkpoint_manager(model, optimizer):
    # setup checkpoints manager
    checkpoint = tf.train.Checkpoint(
      step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=FLAGS.checkpoints_dir, max_to_keep=10
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
    return checkpoint, manager

train_tic = time.time()
def update_train_losses(train_summary_writer, iterations, metrics, decayed_lr):
    global train_tic
    with train_summary_writer.as_default():
      with tf.name_scope("loss_train"):
        tf.summary.scalar('Total loss', 
          metrics.train_loss.result(), step=iterations)

        tf.summary.scalar('Loc loss', 
          metrics.loc.result(), step=iterations)

        tf.summary.scalar('Conf loss', 
          metrics.conf.result(), step=iterations)

        tf.summary.scalar('Mask loss', 
          metrics.mask.result(), step=iterations)

        tf.summary.scalar('Mask IOU loss', 
          metrics.mask_iou.result(), step=iterations)

      with tf.name_scope("norm"):
        tf.summary.scalar('Global Norm', 
          metrics.global_norm.result(), step=iterations)

    if iterations and iterations % FLAGS.print_interval == 0:
        logging.info(
            ("Iteration {}, LR: {}, Total Loss: {:.4f}, B: {:.4f},  "
              "C: {:.4f}, M: {:.4f}, I: {:.4f}, "
              "global_norm:{:.4f} ({:.1f} seconds)").format(
            iterations,
            decayed_lr,
            metrics.train_loss.result(), 
            metrics.loc.result(),
            metrics.conf.result(),
            metrics.mask.result(),
            metrics.mask_iou.result(),
            metrics.global_norm.result(),
            time.time()-train_tic
        ))
        metrics.train_loss.reset_states()
        metrics.loc.reset_states()
        metrics.conf.reset_states()
        metrics.mask.reset_states()
        metrics.mask_iou.reset_states()
        metrics.global_norm.reset_states()
        train_tic = time.time()
        
def update_val_losses(test_summary_writer, iterations, metrics, coco_metrics, config):
    if config.PREDICT_MASK:
        metrics.precision_mAP.update_state(
          coco_metrics['DetectionMasks_Precision/mAP'])
        metrics.precision_mAP_50IOU.update_state(
          coco_metrics['DetectionMasks_Precision/mAP@.50IOU'])
        metrics.precision_mAP_75IOU.update_state(
          coco_metrics['DetectionMasks_Precision/mAP@.75IOU'])
        metrics.precision_mAP_small.update_state(
          coco_metrics['DetectionMasks_Precision/mAP (small)'])
        metrics.precision_mAP_medium.update_state(
          coco_metrics['DetectionMasks_Precision/mAP (medium)'])
        metrics.precision_mAP_large.update_state(
          coco_metrics['DetectionMasks_Precision/mAP (large)'])
        metrics.recall_AR_1.update_state(
          coco_metrics['DetectionMasks_Recall/AR@1'])
        metrics.recall_AR_10.update_state(
          coco_metrics['DetectionMasks_Recall/AR@10'])
        metrics.recall_AR_100.update_state(
          coco_metrics['DetectionMasks_Recall/AR@100'])
        metrics.recall_AR_100_small.update_state(
          coco_metrics['DetectionMasks_Recall/AR@100 (small)'])
        metrics.recall_AR_100_medium.update_state(
          coco_metrics['DetectionMasks_Recall/AR@100 (medium)'])
        metrics.recall_AR_100_large.update_state(
          coco_metrics['DetectionMasks_Recall/AR@100 (large)'])
    else:
        metrics.precision_mAP.update_state(
          coco_metrics['DetectionBoxes_Precision/mAP'])
        metrics.precision_mAP_50IOU.update_state(
          coco_metrics['DetectionBoxes_Precision/mAP@.50IOU'])
        metrics.precision_mAP_75IOU.update_state(
          coco_metrics['DetectionBoxes_Precision/mAP@.75IOU'])
        metrics.precision_mAP_small.update_state(
          coco_metrics['DetectionBoxes_Precision/mAP (small)'])
        metrics.precision_mAP_medium.update_state(
          coco_metrics['DetectionBoxes_Precision/mAP (medium)'])
        metrics.precision_mAP_large.update_state(
          coco_metrics['DetectionBoxes_Precision/mAP (large)'])
        metrics.recall_AR_1.update_state(
          coco_metrics['DetectionBoxes_Recall/AR@1'])
        metrics.recall_AR_10.update_state(
          coco_metrics['DetectionBoxes_Recall/AR@10'])
        metrics.recall_AR_100.update_state(
          coco_metrics['DetectionBoxes_Recall/AR@100'])
        metrics.recall_AR_100_small.update_state(
          coco_metrics['DetectionBoxes_Recall/AR@100 (small)'])
        metrics.recall_AR_100_medium.update_state(
          coco_metrics['DetectionBoxes_Recall/AR@100 (medium)'])
        metrics.recall_AR_100_large.update_state(
          coco_metrics['DetectionBoxes_Recall/AR@100 (large)'])

    with test_summary_writer.as_default():
      with tf.name_scope("loss_val"):
        tf.summary.scalar('V Total loss', 
          metrics.valid_loss.result(), step=iterations)

        tf.summary.scalar('V Loc loss', 
          metrics.v_loc.result(), step=iterations)

        tf.summary.scalar('V Conf loss', 
          metrics.v_conf.result(), step=iterations)

        tf.summary.scalar('V Mask loss', 
          metrics.v_mask.result(), step=iterations)

        tf.summary.scalar('V Mask IOU loss', 
          metrics.v_mask_iou.result(), step=iterations)

      with tf.name_scope("precision"):
        tf.summary.scalar('Precision mAP', 
          metrics.precision_mAP.result(), step=iterations)

        tf.summary.scalar('Precision mAP@.50IOU', 
          metrics.precision_mAP_50IOU.result(), step=iterations)

        tf.summary.scalar('precision mAP@.75IOU', 
          metrics.precision_mAP_75IOU.result(), step=iterations)

        tf.summary.scalar('precision mAP (small)', 
          metrics.precision_mAP_small.result(), step=iterations)

        tf.summary.scalar('precision mAP (medium)', 
          metrics.precision_mAP_medium.result(), step=iterations)

        tf.summary.scalar('precision mAP (large)', 
          metrics.precision_mAP_large.result(), step=iterations)

      with tf.name_scope("recall"):
        tf.summary.scalar('recall AR@1', 
          metrics.recall_AR_1.result(), step=iterations)

        tf.summary.scalar('recall AR@10', 
          metrics.recall_AR_10.result(), step=iterations)

        tf.summary.scalar('recall AR@100', 
          metrics.recall_AR_100.result(), step=iterations)

        tf.summary.scalar('recall AR@100 (small)', 
          metrics.recall_AR_100_small.result(), step=iterations)

        tf.summary.scalar('recall AR@100 (medium)', 
          metrics.recall_AR_100_medium.result(), step=iterations)

        tf.summary.scalar('recall AR@100 (large)', 
          metrics.recall_AR_100_large.result(), step=iterations)

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
                                metrics.train_loss.result(),
                                metrics.loc.result(),
                                metrics.conf.result(),
                                metrics.mask.result(),
                                metrics.mask_iou.result(),
                                metrics.global_norm.result()))
    logging.info(valid_template.format(iterations + 1,
                                metrics.precision_mAP.result(),
                                metrics.precision_mAP_50IOU.result(),
                                metrics.precision_mAP_75IOU.result(),
                                metrics.precision_mAP_small.result(),
                                metrics.precision_mAP_medium.result(),
                                metrics.precision_mAP_large.result(),
                                metrics.recall_AR_1.result(),
                                metrics.recall_AR_10.result(),
                                metrics.recall_AR_100.result(),
                                metrics.recall_AR_100_small.result(),
                                metrics.recall_AR_100_medium.result(),
                                metrics.recall_AR_100_large.result(),
                                metrics.valid_loss.result(),
                                metrics.v_loc.result(),
                                metrics.v_conf.result(),
                                metrics.v_mask.result(),
                                metrics.v_mask_iou.result()))

def add_to_coco_evaluator(valid_labels, output, config , coco_evaluator, _h,_w):
    image_id = int(time.time()*1000000)
    gt_num_box = valid_labels['num_obj'][0].numpy()
    gt_boxes = valid_labels['boxes_norm'][0][:gt_num_box]
    gt_boxes = gt_boxes.numpy()*np.array([_h,_w,_h,_w])
    gt_classes = valid_labels['classes'][0][:gt_num_box].numpy()

    if config.PREDICT_MASK:
      gt_masks = valid_labels['mask_target'][0][:gt_num_box].numpy()

      # gt_masked_image = np.zeros((gt_num_box, _h, _w), dtype=np.uint8)
      # for _b in range(gt_num_box):
      #     box = gt_boxes[_b]
      #     box = np.round(box).astype(int)
      #     (startY, startX, endY, endX) = box.astype("int")
      #     boxW = endX - startX
      #     boxH = endY - startY
      #     if boxW > 0 and boxH > 0:
      #       _m = cv2.resize(gt_masks[_b].astype("uint8"), (boxW, boxH))
      #       gt_masked_image[_b][startY:endY, startX:endX] = _m

      coco_evaluator.add_single_ground_truth_image_info(
          image_id='image'+str(image_id),
          groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes: gt_boxes,
            standard_fields.InputDataFields.groundtruth_classes: gt_classes,
            standard_fields.InputDataFields.groundtruth_instance_masks: gt_masks
          })
    else:
      coco_evaluator.add_single_ground_truth_image_info(
          image_id='image'+str(image_id),
          groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes: gt_boxes,
            standard_fields.InputDataFields.groundtruth_classes: gt_classes,
          })

    det_num = np.count_nonzero(output['detection_scores'][0].numpy()> config.CONF_THRESH)

    det_boxes = output['detection_boxes'][0][:det_num]
    det_boxes = det_boxes.numpy()*np.array([_h,_w,_h,_w])
    det_scores = output['detection_scores'][0][:det_num].numpy()
    det_classes = output['detection_classes'][0][:det_num].numpy().astype(int)

    if config.PREDICT_MASK:
      det_masks = output['detection_masks'][0][:det_num].numpy()
      det_masks = (det_masks > 0.5).astype("uint8")
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
    else:
      coco_evaluator.add_single_detected_image_info(
          image_id='image'+str(image_id),
          detections_dict={
              standard_fields.DetectionResultFields.detection_boxes: det_boxes,
              standard_fields.DetectionResultFields.detection_scores: det_scores,
              standard_fields.DetectionResultFields.detection_classes: det_classes,
          })

def init():
    physical_devices = tf.config.list_physical_devices('GPU')
    if FLAGS.multi_gpu:
      try:
        for i in len(physical_devices):
          tf.config.experimental.set_memory_growth(physical_devices[i], True)
          # tf.config.experimental.set_virtual_device_configuration(physical_devices[i], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=19240)])
      except:
              print("Invalid device or cannot modify virtual devices once initialized.")
              pass
    else:
      try:
              tf.config.experimental.set_memory_growth(physical_devices[0], True)
      except:
              print("Invalid device or cannot modify virtual devices once initialized.")
              pass

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

def main(argv):
    init()
    config = Config()
    config.display()
    mirrored_strategy = tf.distribute.MirroredStrategy()

    if FLAGS.multi_gpu:
        with mirrored_strategy.scope():
            model = MaskED(config)
            add_weight_decay(model, config.WEIGHT_DECAY)   
            optimizer = get_optimizer(config)
            checkpoint, manager = get_checkpoint_manager(model, optimizer)
        BATCH_SIZE_PER_REPLICA = config.BATCH_SIZE
        global_batch_size = (BATCH_SIZE_PER_REPLICA *
                        mirrored_strategy.num_replicas_in_sync)
    else:
        model = MaskED(config)
        add_weight_decay(model, config.WEIGHT_DECAY)   
        optimizer = get_optimizer(config)
        checkpoint, manager = get_checkpoint_manager(model, optimizer)

    # -----------------------------------------------------------------
    # Creating dataloaders for training and validation
    logging.info("Creating the training dataloader from: %s..." % \
      FLAGS.tfrecord_train_dir)
    train_dataset = dataset_coco.prepare_dataloader(
      config, 
      tfrecord_dir=FLAGS.tfrecord_train_dir,
      feature_map_size=model.feature_map_size,
      batch_size=global_batch_size if FLAGS.multi_gpu else config.BATCH_SIZE,
      subset='train')
    if FLAGS.multi_gpu:
        train_dataset_dist = mirrored_strategy.experimental_distribute_dataset(train_dataset)

    logging.info("Creating the validation dataloader from: %s..." % \
      FLAGS.tfrecord_val_dir)
    valid_dataset = dataset_coco.prepare_dataloader(
      config,
      tfrecord_dir=FLAGS.tfrecord_val_dir,
      feature_map_size=model.feature_map_size,
      batch_size=1,
      subset='val')

    criterion = loss.Loss(config)
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
    if config.PREDICT_MASK:
      coco_evaluator = coco_evaluation.CocoMaskEvaluator(
        _get_categories_list(FLAGS.label_map))
    else:
      coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list(FLAGS.label_map))

    metrics = coco_evaluation.Matric()
    best_val = 1e10
    iterations = checkpoint.step.numpy()

    def train_step(image, labels):
        clip_factor=0.01
        eps=1e-3
        with options({'constant_folding': True,
                      'layout_optimize': True,
                      'loop_optimization': True,
                      'arithmetic_optimization': True,
                      'remapping': True}):
            with tf.GradientTape() as tape:
                output = model([image, labels['boxes_norm']], training=True)

                loc_loss, conf_loss, mask_loss, mask_iou_loss, \
                    = criterion(model, output, labels, config.NUM_CLASSES+1, image)

                if FLAGS.multi_gpu:
                    loc_loss = tf.nn.compute_average_loss(loc_loss, global_batch_size=global_batch_size)
                    conf_loss = tf.nn.compute_average_loss(conf_loss, global_batch_size=global_batch_size)
                    mask_loss = tf.nn.compute_average_loss(mask_loss, global_batch_size=global_batch_size)
                    mask_iou_loss = tf.nn.compute_average_loss(mask_iou_loss, global_batch_size=global_batch_size) 
                    total_loss = loc_loss + conf_loss + mask_loss + mask_iou_loss
                else:
                    total_loss = tf.reduce_sum(loc_loss + conf_loss + mask_loss + mask_iou_loss)
                    loc_loss = tf.reduce_sum(loc_loss)
                    conf_loss = tf.reduce_sum(conf_loss)
                    mask_loss = tf.reduce_sum(mask_loss)
                    mask_iou_loss = tf.reduce_sum(mask_iou_loss)
            grads = tape.gradient(total_loss, model.trainable_variables)
            if config.USE_AGC:
              agc_gradients = adaptive_clip_grad(model.trainable_variables, grads, 
                                                  clip_factor=clip_factor, eps=eps)
              optimizer.apply_gradients(zip(agc_gradients, model.trainable_variables))
              metrics.global_norm.update_state(tf.linalg.global_norm(agc_gradients))
            else:
              optimizer.apply_gradients(zip(grads, model.trainable_variables))
              metrics.global_norm.update_state(tf.linalg.global_norm(grads))

            return (loc_loss, conf_loss, mask_loss, mask_iou_loss, total_loss, grads)

    @tf.function
    def distributed_train_step(image, labels):
        per_replica_losses = mirrored_strategy.run(train_step, args=(image, labels,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

    for image, labels in train_dataset:
        # check iteration and change the learning rate
        if iterations > config.TRAIN_ITER:
            break

        checkpoint.step.assign_add(1)
        iterations += 1
        
        if FLAGS.multi_gpu:
            loc_loss, conf_loss, mask_loss, mask_iou_loss, total_loss, grads = distributed_train_step(image, labels)
        else:
            loc_loss, conf_loss, mask_loss, mask_iou_loss, total_loss, grads = train_step(image, labels)
        metrics.train_loss.update_state(total_loss)
        metrics.loc.update_state(loc_loss)
        metrics.conf.update_state(conf_loss)
        metrics.mask.update_state(mask_loss)
        metrics.mask_iou.update_state(mask_iou_loss)
        update_train_losses(train_summary_writer, iterations, metrics, optimizer._decayed_lr(var_dtype=tf.float32))

        if iterations and iterations % FLAGS.save_interval == 0:
            # save checkpoint
            save_path = manager.save()

            logging.info("Saved checkpoint for step {}: {}".format(
              int(iterations), save_path))

            # validation
            valid_iter = 0
            for valid_image, valid_labels in tqdm(valid_dataset, ):
                if valid_iter > FLAGS.valid_iter:
                    break
                # calculate validation loss
                output = model([valid_image, valid_labels['boxes_norm']], training=False)

                valid_loc_loss, valid_conf_loss, valid_mask_loss, \
                valid_mask_iou_loss = \
                criterion(model, output, valid_labels, config.NUM_CLASSES+1)

                valid_total_loss = sum(valid_loc_loss + valid_conf_loss + valid_mask_loss + valid_mask_iou_loss)
                metrics.valid_loss.update_state(valid_total_loss)
                metrics.v_loc.update_state(valid_loc_loss)
                metrics.v_conf.update_state(valid_conf_loss)
                metrics.v_mask.update_state(valid_mask_loss)
                metrics.v_mask_iou.update_state(valid_mask_iou_loss)
                
                add_to_coco_evaluator(valid_labels, output, config , coco_evaluator, valid_image.shape[1], valid_image.shape[2])
                valid_iter += 1

            coco_metrics = coco_evaluator.evaluate()
            update_val_losses(test_summary_writer, iterations, metrics, coco_metrics, config)
            coco_evaluator.clear()
            metrics.reset()

if __name__ == '__main__':
    app.run(main)
