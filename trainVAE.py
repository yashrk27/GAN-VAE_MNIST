# Python Imports
import numpy as np
import tensorflow as tf
import time

# Internal Imports
import modelVAE
from utils import Utils


def log_normal_pdf(sample, mean, log_var, raxis=1):
    # Calculates the negative log normal probability dist function
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + log2pi), axis=raxis)


def modelLoss(model, x):
    # Calculating the loss for model
    mean, log_var = model.encode(x)
    z = model.reparameterize(mean, log_var)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, log_var)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def trainStep(model, x, optimizer):
    # Step function for each epoch
    with tf.GradientTape() as tape:
        loss = modelLoss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train():
    global epoch
    for epoch in range(1, epochs + 1):
        # Initialising the time for each epoch
        initial = time.time()
        for train_x in train_dataset:
            trainStep(model, train_x, optimizer)
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(modelLoss(model, test_x))
        print('Training time for ', epoch, 'is', time.time() - initial, 'sec')
        Utils.saveVAEImage(model, epoch, test_sample, "output/VAE")


# Loading the MNIST data
(trainImg, _), (testImg, _) = tf.keras.datasets.mnist.load_data()
trainImg = Utils.preprocessVAE(trainImg)
testImg = Utils.preprocessVAE(testImg)

# Setting the train, test and batch size
trainSize = 60000
batchSize = 32
testSize = 10000
# Loading the data into batches
train_dataset = (tf.data.Dataset.from_tensor_slices(trainImg).shuffle(trainSize).batch(batchSize))
test_dataset = (tf.data.Dataset.from_tensor_slices(testImg).shuffle(testSize).batch(batchSize))

# Initializing optimiser for model
optimizer = tf.keras.optimizers.Adam(1e-4)

# Setting the epochs for training
epochs = 50
numOutputImg = 16

model = modelVAE.VAE(2)

for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:numOutputImg, :, :, :]

train()
