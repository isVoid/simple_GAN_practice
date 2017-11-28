from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
from ops import *
from utils import *

from random import shuffle

USE_TENSORBOARD = False

class GAN(object):

    def __init__(self, args):
        self.learning_rate = args.get("learning_rate", 1e-4)
        self.mini_batch_size = args.get("mini_batch_size", 256)
        self.epochs = args.get("epochs", -1)
        self.k = args.get("k", 1)

    def _hp_string(self, epoch):
        return "k=%s_lr=%s_mbs=%s_e=%s" % (self.k, self.learning_rate,
            self.mini_batch_size, epoch)

    def generator(self, z):
        with tf.variable_scope("Generator"):

            a1 = dense_bn_lrelu(z, 1024, name = "L1")
            a2 = dense_bn_lrelu(a1, 128 * 7 * 7, name = "L2")
            a2_2d = tf.reshape(a2, [self.mini_batch_size, 7, 7, 128])

            conv_t3 = conv_tran_bn_lrelu(a2_2d, [self.mini_batch_size, 14, 14, 64], name="conv_t3")
            out = conv_tran_tanh(conv_t3, [self.mini_batch_size, 28, 28, 1], name = "conv_t4")

            return out

    def discriminator(self, x, reuse = False):
        with tf.variable_scope("Discriminator", reuse = reuse):

            conv1 = conv_lrelu(x, 1, 64, name = "conv1")
            conv2 = conv_bn_lrelu(conv1, 64, 128, name = "conv2")
            conv2_flat = tf.reshape(conv2, [self.mini_batch_size, -1])

            fc3 = dense_bn_lrelu(conv2_flat, 1024, name = "fc3")
            prob = dense_sigmoid(fc3, 1, name = "prob")

            return prob

    def model(self):

        z = tf.placeholder(tf.float32, shape = (None, 100), name = "noise")

        X = tf.placeholder(tf.float32, shape = (None, 28, 28, 1), name = "X")

        # Build Model
        Gz = self.generator(z)
        DGz = self.discriminator(Gz, reuse = False)
        Dx = self.discriminator(X, reuse = True)

        print ("Gz shape", Gz.shape)
        print ("DGz shape", DGz.shape)
        print ("Dx shape", Dx.shape)

        return z, X, Gz, DGz, Dx


    def train(self, X_train):

        tf.reset_default_graph()

        with tf.device("/gpu:0"):

            z, X, Gz, DGz, Dx = self.model()
            
            if USE_TENSORBOARD:
                tf.summary.image("Generated digits", gen_z, 5)
                gen_z = tf.reshape(Gz, shape = [-1, 28, 28, 1])

            d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx))) + \
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DGz, labels = tf.zeros_like(DGz)))

            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DGz, labels = tf.ones_like(DGz)))


        if USE_TENSORBOARD: tf.summary.scalar("d_loss", d_loss)
        if USE_TENSORBOARD: tf.summary.scalar("g_loss", g_loss)

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Discriminator/")
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator/")

        d_optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate,
            name = "d_optim").minimize(d_loss, var_list = d_vars)

        g_optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate,
            name = "g_optim").minimize(g_loss, var_list = g_vars)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
            sess.run(init)
            if USE_TENSORBOARD:
                summ = tf.summary.merge_all()
                writer = tf.summary.FileWriter("log/")
                writer.add_graph(sess.graph)

            cur_k = 0
            seed = 0
            n=0
            X_batches = random_mini_batches(X_train, self.mini_batch_size, seed = 231)
            for cur_epoch in range(self.epochs):
                shuffle(X_batches)
                for i in range(len(X_batches)):
                    noise = np.random.normal(size = (self.mini_batch_size, 100)).astype('float32')
                    mini_x = X_batches[i]
                    if USE_TENSORBOARD:
                        dl, s, _ = sess.run([d_loss, summ, d_optim], feed_dict = {X: mini_x, z: noise})
                    else:
                        dl, _ = sess.run([d_loss, d_optim], feed_dict = {X: mini_x, z: noise})
                    gl, gen, _ = sess.run([g_loss, Gz, g_optim], feed_dict = {z: noise})

                    if USE_TENSORBOARD:
                        writer.add_summary(s, global_step = cur_epoch)
                    print ("Current step: %d, d_loss: %f, g_loss: %f" % (n, dl, gl))
                    n += 1
                gen = inv_norm_transform(gen)
                img = gen_to_img(gen)
                (Image.fromarray(img)).convert('L').save("output/epoch_%s.png" % (cur_epoch))
                saver.save(sess, "model/" + self._hp_string(cur_epoch))
            sess.close()


from tensorflow.examples.tutorials.mnist import input_data
def main(args):
    mnist = input_data.read_data_sets("mnist/", one_hot=True)
    param = {
        "learning_rate" : 1e-5,
        "epochs" : 50000,
        "mini_batch_size" : 128,
        "k" : 1
    }
    g = GAN(param)
    mnist2d = mnist.test.images.reshape((-1, 28, 28, 1))
    mnist2d = norm_transform(mnist2d)
    print (mnist2d.shape)
    g.train(mnist2d)
    # g.train(mnist.train.images)


if __name__ == '__main__':
    main(None)
