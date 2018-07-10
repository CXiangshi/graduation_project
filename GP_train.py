import time
from datetime import datetime


import numpy as np 
import tensorflow as tf 
import GP_model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('trian_dir', '/')
tf.app.flags.DEFINE_integer('max_steps', 100000)
tf.app.flags.DEFINE_boolean('log_device_placement', False)
tf.app.flags.DEFINE_integer('log_frequency', 10)


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            images, labels = GP_model.distorted_inputs()

            logits = GP_model.inference(images)

            loss = GP_model.loss(logits, labels)

            train_op = GP_model.train(loss, global_step)

        class LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self.step = -1
                self.start_time = time.time()
            
            def before_run(self, run_context):
                self.step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self.step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self.start_time
                    self.start_time = current_time

                    loss_value = run_values.results
                    examples_per_sex = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f''sec/batch')
                    print (format_str % (datetime.now(), self.step, examples_per_sex, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
        tf.train.NanTensorHook(loss), LoggerHook()],
        config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:

        while not mon_sess.should_stop():
            mon_sess.run(train_op)
    

if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
tf.gfile.MakeDirs(FLAGS.train_dir)
train()

'''
if __name__== '__main__':
    tf.app.run()
'''