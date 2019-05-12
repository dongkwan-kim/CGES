import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
try:
    from .mnist_model import mnist_conv
    from .utils import comp, cost
    from .cges import cges
except ModuleNotFoundError:
    from mnist_model import mnist_conv
    from utils import comp, cost
    from cges import cges

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


#######################
# Model Configuration #
#######################
tf.app.flags.DEFINE_float('base_lr', 0.05, 'initialized learning rate')
tf.app.flags.DEFINE_float('stepsize', 5000, '')
tf.app.flags.DEFINE_float('decay_rate', 0.9, '')
tf.app.flags.DEFINE_float('memory_usage', 0.94, '')
tf.app.flags.DEFINE_integer('train_display', 100, '')
tf.app.flags.DEFINE_integer('test_iter', 1000, '')
tf.app.flags.DEFINE_integer('max_iter', 30000, '')

#############################
# Regularizer Configuration #
#############################
tf.app.flags.DEFINE_float('lamb', 0.00006, 'regularizer parameter')
tf.app.flags.DEFINE_boolean('cges', True, 'Combined group and exclusive sparsity')

######################
# CGES Configuration #
######################
tf.app.flags.DEFINE_float('mu', 0.8, 'initialized group sparsity ratio')
tf.app.flags.DEFINE_float('chvar', 0.2, '\'mu\' change per layer')

FLAGS = tf.app.flags.FLAGS
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X = tf.placeholder(tf.float32, shape=[None, 784])  # single flattened 28 * 28 pixel MNIST image
Y = tf.placeholder(tf.float32, shape=[None, 10])  # 10 classes output
keep_prob = tf.placeholder(tf.float32)


batch = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    FLAGS.base_lr,  # Base learning rate.
    batch,  # Current index.
    FLAGS.stepsize,  # Decay iteration step.
    FLAGS.decay_rate,  # Decay rate.
    staircase=True)


y_conv = mnist_conv(X, 10, keep_prob)
y_preds = tf.nn.softmax(y_conv)

ff_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=Y))
S_vars = [svar for svar in tf.trainable_variables() if 'weight' in svar.name]

lamb = FLAGS.lamb
mu = FLAGS.mu
chvar = FLAGS.chvar

if not FLAGS.cges:
    ff_loss_reg = ff_loss + learning_rate * 0.01 * \
                  tf.reduce_sum([tf.nn.l2_loss(var) for var in S_vars])
    cges_op_list = []
else:
    ff_loss_reg = ff_loss
    cges_op_list = cges(learning_rate, lamb, mu, chvar,
                        group_layerwise=[1., 1.0, 1. / 15, 1. / 144],
                        exclusive_layerwise=[1., 0.5, 15., 144.],
                        variable_filter=lambda name: "weight" in name)


opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(ff_loss_reg, global_step=batch)
correct_prediction = tf.equal(tf.argmax(y_preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

save_sparsity = []
_sp = []
for i in range(FLAGS.max_iter):
    batch = mnist.train.next_batch(100)

    # Display
    if (i + 1) % FLAGS.train_display == 0:
        train_accuracy, tr_loss = sess.run([accuracy, ff_loss],
                                           feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
        print("step %d, lr %.4f, training accuracy %g"
              % (i + 1, sess.run(learning_rate), train_accuracy))

        ratio_w, sp = comp(S_vars)
        _sp = sess.run(sp)

        print("loss: %.4f sp: %0.4f %0.4f %0.4f %0.4f :: using param : %.4f"
              % (tr_loss, _sp[0], _sp[1], _sp[2], _sp[3], sess.run(ratio_w)))

    # Training
    sess.run(opt, feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})
    if FLAGS.cges:
        _ = sess.run(cges_op_list)

    # Testing
    if (i + 1) % FLAGS.test_iter == 0:
        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                 Y: mnist.test.labels,
                                                 keep_prob: 1.0})
        print("test accuracy %0.4f" % test_acc)

        # Computing FLOP
        flop = cost(_sp)
        print("FLOP : %.4f" % flop)
        if FLAGS.cges:
            print('CGES, lambda : %f, mu : %.2f, chvar : %.2f' % (lamb, mu, chvar))
