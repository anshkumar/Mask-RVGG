import tensorflow as tf
from model import MaskED
from config import Config
from tensorflow.keras import layers

class Model(MaskED):
    def __init__(self, config):
        super(Model, self).__init__(config)
    
    @tf.function()
    def call(self, inputs, training=False):
        if self.config.BACKBONE == 'resnet50':
            inputs = tf.keras.applications.resnet50.preprocess_input(inputs)
        features = self.backbone(inputs, training=False)
        if not self.config.USE_FPN:
            classification = self.class_net(features)
            classification = layers.Concatenate(axis=1, name='classification')(classification)
            regression = self.box_net(features)
            regression = layers.Concatenate(axis=1, name='regression')(regression)
        else:
            # Prediction Head branch
            pred_cls = []
            pred_offset = []
            # all output from FPN use same prediction head
            for f_map in features:
                cls, offset = self.predictionHead(f_map)
                pred_cls.append(cls)
                pred_offset.append(offset)
            classification = tf.concat(pred_cls, axis=1)
            regression = tf.concat(pred_offset, axis=1)
        pred = {
            'regression': regression,
            'classification': classification,
            'priors': self.priors
        }
        pred.update(self.detect(pred))
        if self.config.PREDICT_MASK:
            masks = self.mask_head(pred['detection_boxes'],
                            features[:-2],
                            self.num_classes,
                            self.config,
                            training)
            pred.update({'detection_masks': masks})
        return pred

config = Config()
model = Model(config)
# model.call = call
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore('checkpoints/ckpt-1')

_ = model(tf.random.uniform(shape=[16,512,512,3], minval=0, maxval=255, dtype=tf.float32), training=False)
model.save('test')
