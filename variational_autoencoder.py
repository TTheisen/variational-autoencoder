import tensorflow as tf 
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
tf.reset_default_graph()

'''
Variational AutoEncoders:

--> probabilistic autoencoder, outputs are partly determined via randomness
--> generative autoencoder, generate new instances that look like they are sampled from the training set

1. --> Instead of producing a coding (central hidden layer in stacked, symmetrical architecture) for a given input
the encoder will produce a mean coding *u* and standard deviation *sigma*. 
2. --> The new coding is produced my sampling from *u* and *sigma*.
3. --> The decoder then decodes this coding.

During training the cost function pushes the codings to migrate within the latent space and occupy a roughly
hyper spherical region. 

The cost function is composed of two parts: 
    --> reconstruction loss: pushes the autoencoder/decoder to reproduce its inputs
    --> latent loss: pushes the autoencoder to have codings look as though they are sampled from a gaussian distribution
        -> use Kullback-Leibler Divergence
'''

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_TRAIN_SAMPLES = x_train.shape[0]
eps = 1e-10

n_inputs = 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.0001

initializer = tf.contrib.layers.variance_scaling_initializer()

dense_layers = partial(
    tf.layers.dense,
    activation = tf.nn.elu,
    kernel_initializer = initializer
)

X = tf.placeholder(tf.float32, shape = [None, n_inputs])
hidden1 = dense_layers(X, n_hidden1)
hidden2 = dense_layers(hidden1, n_hidden2)
hidden3_mean = dense_layers(hidden2, n_hidden3, activation = None)
hidden3_gamma = dense_layers(hidden2, n_hidden3, activation = None)
noise = tf.random_normal(tf.shape(hidden3_gamma), dtype = tf.float32)
hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise
hidden4 = dense_layers(hidden3, n_hidden4)
hidden5 = dense_layers(hidden4, n_hidden5)
logits = dense_layers(hidden5, n_outputs, activation = None)
outputs = tf.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)
latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_operation = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 25
n_digits = 10
batch_size = 150

with tf.Session() as sess:
    init.run() 
    for epoch in range(n_epochs):
        print(f"Current Epoch: {epoch}")
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size) 
            sess.run(training_operation, feed_dict={X: X_batch})
            loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
    random_codings = np.random.normal(size = [n_digits, n_hidden3])
    output_digits = outputs.eval(feed_dict={hidden3: random_codings})


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

plt.figure(figsize=(8,50))
for i in range(n_digits):
    plt.subplot(n_digits, 10, i + 1)
    plot_image(output_digits[i])
plt.show()
