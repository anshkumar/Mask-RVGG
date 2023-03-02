import tensorflow as tf
from model import MaskRVGG
from absl import app
from absl import flags
from importlib.machinery import SourceFileLoader

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint', None,
                    'checkpoint path to export.')
flags.DEFINE_string('config', None,
                    'config path to export.')
flags.DEFINE_string('out_saved_model_dir', None,
                    'saved_model directory to save')

class Model(MaskRVGG):
    def __init__(self, config, imagenet_path='',  base_model=None, deploy=False):
        super(Model, self).__init__(config, imagenet_path, base_model, deploy)

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

def main(argv):
    cnf = SourceFileLoader("", FLAGS.config).load_module()
    
    config = cnf.Config()
    model = Model(config)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(FLAGS.checkpoint)

    _ = model(tf.random.uniform(shape=[1]+config.IMAGE_SHAPE, minval=0, maxval=255, dtype=tf.float32), training=False)
    # model.save(FLAGS.out_saved_model_dir)

    deploy_model = Model(config, base_model=model, deploy=True)
    checkpoint = tf.train.Checkpoint(model=deploy_model)
    status = checkpoint.restore(FLAGS.checkpoint)

    deploy_y = deploy_model(tf.random.uniform(shape=[1]+config.IMAGE_SHAPE, minval=0, maxval=255, dtype=tf.float32), training=False)

    deploy_model.save(FLAGS.out_saved_model_dir)

if __name__ == '__main__':
    app.run(main)
