import tensorflow as tf
from model import MaskRVGG
from config import Config

class Model(MaskRVGG):
    def __init__(self, config, base_model=None, deploy=False):
        super(Model, self).__init__(config, base_model, deploy)

    @tf.function()
    def call(self, inputs, training=False):
        if self.config.BACKBONE == 'resnet50':
            inputs = tf.keras.applications.resnet50.preprocess_input(inputs)
        features = self.backbone(inputs, training=False)

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

test_inp = tf.random.uniform(shape=[16,512,512,3], minval=0, maxval=255, dtype=tf.float32)

config = Config()
model = Model(config)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore('checkpoints/ckpt-114').expect_partial()
train_y = model(test_inp, training=False)
# model.save('saved_models')

deploy_model = Model(config, base_model=model, deploy=True)
checkpoint = tf.train.Checkpoint(model=deploy_model)
status = checkpoint.restore('checkpoints/ckpt-114').expect_partial()

deploy_y = deploy_model(test_inp, training=False)

deploy_model.save('saved_models')
