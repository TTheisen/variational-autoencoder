import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")
tf.reset_default_graph()

'''
Variational AutoEncoders:

Consist of an encoder, decoder, and two-part loss function.

x: input data
z: hidden/latent space that the encoder maps x to (the encoder learns an efficient
    compression of x in lower dimensional space)

This process can be represented as q(z | x) where q is the encoder. This lower dimensional 
    space is stochastic. Samples can be drawn from q(z | x). 

The decoder takes z as input, which can be represented as  p(x* | z). The probability distribution
    of a single pixel can then be presented using a Bernoulli distribution.

p is a standard normal distribution with unit variance. If the encoder outputs representations z that are different
    than those from N(0,1) it will receive a penalty. This regularization term is used to keep the representation of z
    significantly diverse for each digit. 

--> probabilistic autoencoder, outputs are partly determined via randomness
--> generative autoencoder, generate new instances that look like they are sampled from the training set

1. --> Instead of producing a coding (central hidden layer in stacked, symmetrical architecture) for a given input
the encoder will produce a mean coding *u* and standard deviation *sigma*. 
2. --> The new coding is produced my sampling from *u* and *sigma*.
3. --> The decoder then decodes this coding.

During training the cost function pushes the codings to migrate within the latent space and occupy a roughly
hyper spherical region. 

The cost function is composed of two parts: 
    --> reconstruction loss: pushes the autoencoder to reproduce its inputs (expected negative log-likelihood)
        -> if the decoder output doesn't reconstruct the data well, statistically the decoder parameterizes a likelihood distribution that
            doesn't resemble the probability mass distribution of the true data
    --> latent loss: pushes the autoencoder to have codings look as though they are sampled from a gaussian distribution
        -> use Kullback-Leibler Divergence
'''

def xavier_initialization(dim_in, dim_out, constant = 1):
    total = dim_in + dim_out
    high = constant * np.sqrt(6.0 / total) 
    low = -1 * high
    return tf.random_uniform((dim_in, dim_out),
            minval = low, maxval = high, dtype = tf.float32)

class VAE:
    def __init__(self, network_architecture, lr = 0.01, bs = 32, e = 1, m = 1e-9):
        self.network_architecture = network_architecture
        self.learning_rate = lr
        self.batch_size = bs
        self.epochs = e
        self.margin = m
        self.x = tf.placeholder(tf.float32, shape = [None, self.network_architecture['n_input']])
        self.build()
        self.loss_optimizer()
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    def init_weights(self, encoder_n_hidden1, encoder_n_hidden2, decoder_n_hidden1, decoder_n_hidden2, n_input, n_z):
        weights = dict()
        weights['weights_encoder'] = {
            'hidden1': tf.Variable(xavier_initialization(n_input, encoder_n_hidden1)),
            'hidden2': tf.Variable(xavier_initialization(encoder_n_hidden1, encoder_n_hidden2)),
            'out_mu': tf.Variable(xavier_initialization(encoder_n_hidden2, n_z)),
            'out_sigma': tf.Variable(xavier_initialization(encoder_n_hidden2, n_z))
        }
        weights['biases_encoder'] = {
            'bias1': tf.Variable(tf.zeros([encoder_n_hidden1], dtype=tf.float32)),
            'bias2': tf.Variable(tf.zeros([encoder_n_hidden2], dtype=tf.float32)),
            'out_mu': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }
        weights['weights_decoder'] = {
            'hidden1': tf.Variable(xavier_initialization(n_z, decoder_n_hidden1)),
            'hidden2': tf.Variable(xavier_initialization(decoder_n_hidden1, decoder_n_hidden2)),
            'out_mu': tf.Variable(xavier_initialization(decoder_n_hidden2, n_input)),
            'out_sigma': tf.Variable(xavier_initialization(decoder_n_hidden2, n_input))
        }
        weights['biases_decoder'] = {
            'bias1': tf.Variable(tf.zeros([decoder_n_hidden1], dtype=tf.float32)),
            'bias2': tf.Variable(tf.zeros([decoder_n_hidden2], dtype=tf.float32)),
            'out_mu': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        }
        return weights  
    def build(self):
        network_weights = self.init_weights(**self.network_architecture)
        self.z_mean, self.z_sigma = self.encoder_network(network_weights['weights_encoder'], network_weights['biases_encoder'])
        n_z = self.network_architecture['n_z']
        epsilon = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32) #100 * 20
        ''' z = mu + sigma * epsilon '''
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_sigma)), epsilon)) 
        self.x_reconstruction_mean = self.decoder_network(network_weights['weights_decoder'], network_weights['biases_decoder'])
    def encoder_network(self, weights, biases):
        layer1 = tf.nn.softplus(tf.add(tf.matmul(self.x, weights['hidden1']), biases['bias1']))
        layer2 = tf.nn.softplus(tf.add(tf.matmul(layer1, weights['hidden2']), biases['bias2']))
        z_mean = tf.add(tf.matmul(layer2, weights['out_mu']), biases['out_mu'])
        z_sigma = tf.add(tf.matmul(layer2, weights['out_sigma']), biases['out_sigma'])
        return (z_mean, z_sigma)
    def decoder_network(self, weights, biases):
        layer1 = tf.nn.softplus(tf.add(tf.matmul(self.z, weights['hidden1']), biases['bias1']))
        layer2 = tf.nn.softplus(tf.add(tf.matmul(layer1, weights['hidden2']), biases['bias2']))
        reconstructed_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['out_mu']), biases['out_mu']))
        return reconstructed_mean
    def loss_optimizer(self):
        reconstruction_loss = -1 * tf.reduce_sum(self.x * tf.log(self.margin + self.x_reconstruction_mean) + \
                              (1 - self.x) * tf.log(self.margin + 1 - self.x_reconstruction_mean), 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_sigma - tf.square(self.z_mean) - tf.exp(self.z_sigma), 1)
        self.cost = tf.reduce_mean(reconstruction_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
    def fit(self, X):
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
        return opt, cost
    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    def generate(self, z_mu=None):
        if z_mu is None: 
            z_mu = np.random.normal(size = self.network_architecture['n_z'])
        return self.sess.run(self.x_reconstruction_mean, feed_dict={self.z: z_mu})
    def reconstruct(self, X):
        return self.sess.run(self.x_reconstruction_mean, feed_dict={self.x: X})

def train(network_architecture, lr=0.001, bs=100, e=50):
    vae = VAE(network_architecture, lr = lr, bs = bs)
    samples = mnist.train.num_examples 
    for epoch in range(e):
        mean_cost = 0.0
        total_batches = samples // bs
        for _ in range(total_batches):
            x_batch, _ = mnist.train.next_batch(bs)
            _, cost = vae.fit(x_batch)
            mean_cost = mean_cost + cost / samples * bs
        print(f"Epoch: {epoch}, Cost: {cost}")
    return vae

network_architecture = dict(encoder_n_hidden1 = 500, encoder_n_hidden2 = 250, 
                            decoder_n_hidden1 = 250, decoder_n_hidden2 = 500,
                            n_input = 28*28, n_z = 2)

vae = train(network_architecture)

x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)

plt.figure(figsize = (8,12))
plt.imshow(x_sample[0].reshape([28,28]))
plt.show()
plt.imshow(x_reconstruct[0].reshape([28,28]))
plt.show()


x_sample, y_sample = mnist.test.next_batch(10000)
print(y_sample)
z_mu = vae.transform(x_sample)
plt.scatter(z_mu[:,0], z_mu[:,1], c = y_sample)
plt.grid()
plt.show()

nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi]]*vae.batch_size)
        x_mean = vae.generate(z_mu)
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))        
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.tight_layout()
plt.show()