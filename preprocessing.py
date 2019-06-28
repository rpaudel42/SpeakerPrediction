# ******************************************************************************
# preprocessing.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/27/19   Paudel     Initial version,
# ******************************************************************************

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from numpy import loadtxt, array

class Preprocessing():
    def __init__(self):
       pass

    def read_test_file(self, block_file):
        '''
        Read test data from the file and store into time series blocks
        :param block_file:
        :return:
        '''
        file = open(block_file, "r")
        speaker_index = 0
        block_index = 0
        block = list()
        blocks = list()
        for line in file.readlines():
            if line == '\n':
                blocks.append(block)
                block_index += 1
                block = list()
            else:
                point_in_time = list()
                line = line.strip('\n')
                for x in line.split(' ')[:12]:
                    point_in_time.append(float(x))
                block.append(point_in_time)
        return blocks

    def read_train_file(self, block_file, block_label_file):
        '''
        Read data from txt file into time series blocks (sequences)
        :param block_file:
        :param block_label_file:
        :return:
        '''
        block_label_index = loadtxt(block_label_file, delimiter=" ").tolist()
        file = open(block_file, "r")
        speaker_index = 0
        block_index = 0
        block = list()
        blocks = list()
        labels = list()
        for line in file.readlines():
            if line == '\n':
                label = list()
                blocks.append(block)
                label.append(speaker_index)
                labels.append(label)
                block_index += 1
                block = list()
                if speaker_index <= 8 and block_index == block_label_index[speaker_index]:
                    speaker_index += 1
                    block_index = 0
            else:
                point_in_time = list()
                line = line.strip('\n')
                for x in line.split(' ')[:12]:
                    point_in_time.append(float(x))
                block.append(point_in_time)
        return blocks, labels

    def pad_to_fixed_size_blocks(self, data_block, max_length, final_block_size):
        '''
        First pad last row till max length, then truncate it to fixed length size
        :param data_block:
        :return:
        '''
        # Padding the sequence with the values in last row to max length
        fixed_size_block = []
        for block in data_block:
            block_len = len(block)
            last_row = block[-1]
            n = max_length - block_len

            to_pad = np.repeat(block[-1], n).reshape(12, n).transpose()
            new_block = np.concatenate([block, to_pad])
            fixed_size_block.append(new_block)

        final_dataset = np.stack(fixed_size_block)

        # truncate the sequence to final_block_size
        final_dataset = pad_sequences(final_dataset, maxlen=final_block_size, padding='post', dtype='float', truncating='post')

        return final_dataset

    def convert_to_vectors(self, data_block, block_label, final_block_size):
        '''
        Convert fixed size block to feature vectors for ML algorithms
        :param data_block:
        :param block_label:
        :param final_block_size:
        :return:
        '''
        block_label = [i[0] for i in block_label]
        # print(block_label)
        vectors = list()
        n_features = 12
        for i in range(len(data_block)):
            block = data_block[i]
            vector = list()
            for row in range(1, final_block_size+1):
                for col in range(n_features):
                    vector.append(block[-row, col])

            vector.append(block_label[i])
            vectors.append(vector)
        vectors = array(vectors)
        vectors =vectors.astype('float32')
        return vectors
