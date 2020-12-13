# Python Imports
import tensorflow as tf
import time

# Internal Imports
from modelGAN import Model
from utils import Utils

# Loading the MNIST data
(trainData, _), (_, _) = tf.keras.datasets.mnist.load_data()

trainData = Utils.preprocess(trainData)

batchSize = 256
batchData = tf.data.Dataset.from_tensor_slices(trainData).shuffle(10000).batch(batchSize)

# Initialising the model
generator = Model.Generator()
discriminator = Model.Discriminator()

# Initialising the optimiser
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Setting the number of epochs for training
numEpoch = 100

noise = 100
outputImgSize = 16
seed = tf.random.normal([outputImgSize, noise])

# Training
for epoch in range(numEpoch):
    # Initialising the time for each epoch
    initial = time.time()
    for batchImg in batchData:
        noiseVal = tf.random.normal([batchSize, noise])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noiseVal, training=True)

            realClassification = discriminator(batchImg, training=True)
            fakeClassification = discriminator(generated_images, training=True)

            loss_G = Model.loss_G(fakeClassification)
            loss_D = Model.loss_D(realClassification, fakeClassification)

        genGradient = gen_tape.gradient(loss_G, generator.trainable_variables)
        disGradient = disc_tape.gradient(loss_D, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(genGradient, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disGradient, discriminator.trainable_variables))
    Utils.saveImage(generator, epoch + 1, seed, "output/GAN/")
    print('Training time for ', epoch, 'is', time.time() - initial, 'sec')
