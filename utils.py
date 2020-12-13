# python import
import matplotlib.pyplot as plt
import os
import numpy as np


class Utils:
    @staticmethod
    def saveImage(model, epoch, test_input, path):
        predictions = model(test_input, training=False)
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray_r')
        plt.savefig(os.path.join(path, 'epoch{:d}.png'.format(epoch)))

    @staticmethod
    def preprocess(data):
        data = data.reshape(data.shape[0], 28, 28, 1).astype('float32')
        # Normalising the data
        data = (data - 127.5) / 127.5
        return data

    @staticmethod
    def preprocessVAE(dataset):
        dataset = dataset.reshape((dataset.shape[0], 28, 28, 1)) / 255.
        return np.where(dataset > .5, 1.0, 0.0).astype('float32')

    @staticmethod
    def saveVAEImage(model, epoch, test_sample, path):
        mean, logvar = model.encode(test_sample)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray_r')

        plt.savefig(os.path.join(path, 'epoch{:d}.png'.format(epoch)))

