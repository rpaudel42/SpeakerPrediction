# ******************************************************************************
# exploration.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/27/19   Paudel     Initial version,
# ******************************************************************************
import numpy as np
from numpy import array, vstack
from numpy.linalg import lstsq
import pandas as pd

# To create plots
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D


class Visualization():
    def visualize_lpc_distribution(self, train_blocks):
        '''
        Visualize all 12 lpc coefficients distribution over all blocks
        :param train_blocks:
        :return:
        '''
        point_in_time = vstack(train_blocks)
        plt.figure(figsize=(10, 25))
        plt.title('LPC coefficients  Distribution')
        coefficients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for c in coefficients:
            plt.subplot(len(coefficients), 1, c + 1)
            plt.hist(point_in_time[:, c], bins=100)
        plt.savefig('fig/lpc_coeff_dist.png')
        # plt.show()

    def visualize_block_length_distribution(self, train_blocks):
        '''
        visualize the distribution of block length
        :param train_blocks:
        :return:
        '''
        points_in_time = [len(x) for x in train_blocks]
        plt.title('Block Length Distribution')
        plt.hist(points_in_time, bins=25)
        plt.savefig('fig/block_len_dist.png')
        # plt.show()

    def lpc_scatter_plot(self, final_train_data):
        '''
        use scatter plot to visualize the grouping of users
        :param final_train_data:
        :return:
        '''
        train_X, train_y = final_train_data[:, :-1], final_train_data[:, -1]
        colormap = get_cmap('viridis')
        colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1 / (6 - 1))]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        ax.scatter(train_X[:, 0], train_X[:, 1], train_X[:, 2], c=train_y, s=50, cmap=mat.colors.ListedColormap(colors))
        plt.title('Speaker Plot')
        plt.savefig('fig/scatter_plot.png')
        # plt.show()

    def visualize_lpc_time_series(self, speaker_blocks):
        '''
        visualize a block of lpc series for each speaker
        :param speaker_blocks:
        :return:
        '''
        # group sequences by speaker
        speakers = [i + 1 for i in range(0,9)]
        speakers_voice = {}
        for speaker in speakers:
            speakers_voice[speaker] = [speaker_blocks[j] for j in range(len(speakers)) if speakers[j] == speaker]
        plt.figure(figsize=(10, 35))
        plt.title('LPC trend for each speaker')
        for i in speakers:
            plt.subplot(len(speakers), 1, i)
            coeff_series = vstack(speakers_voice[i][0])
            for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                plt.plot(coeff_series[:, j], label='test')
            plt.title('Speaker ' + str(i), y=0, loc='left')
        plt.savefig('fig/lpc_series.png')
        # plt.show()

    def regress(self, y):
        X = array([i for i in range(len(y))]).reshape(len(y), 1)
        b = lstsq(X, y)[0][0]
        yhat = b * X[:, 0]
        return yhat

    def visualize_fitted_lpc_series(self, speaker_blocks):
        '''
        visualize a fitted lpc series for each speaker to see the trend
        :param speaker_blocks:
        :return:
        '''
        speakers = [i + 1 for i in range(0,9)]
        speakers_voice = {}
        for speaker in speakers:
            speakers_voice[speaker] = [speaker_blocks[j] for j in range(len(speakers)) if speakers[j] == speaker]
        plt.figure(figsize=(10, 25))
        plt.title('LPC trend for each speaker')
        for i in speakers:
            plt.subplot(len(speakers), 1, i)
            coeff_series = vstack(speakers_voice[i][0])
            plt.plot(coeff_series[:, i])
            plt.plot(self.regress(coeff_series[:, i]))
            plt.title('Speaker ' + str(i), y=0, loc='left')
        plt.savefig('fig/fitted_lpc_series.png')
        # plt.show()

    def show_training_history(self, history):
        '''
        Show LSTM model training loss over each epoch
        :param history:
        :return:
        '''
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'])
        plt.title('LSTM model training loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('fig/lstm_train_error.png')
        # pyplot.show()