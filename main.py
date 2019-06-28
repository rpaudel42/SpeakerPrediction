# ******************************************************************************
# main.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/27/19   Paudel     Initial version,
# *****************************************************************************

import argparse
from preprocessing import Preprocessing
from exploration import Visualization
from models import Models
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args():
    '''
    :return: all command line arguments read
    '''
    args = argparse.ArgumentParser("TrueMotionProject")

    args.add_argument('-t','--train_data', default='data/train.txt',
                      help='Path to train data file')

    args.add_argument("-l","--train_label", default = "data/train_block_labels.txt",
                      help="Path to training labels file")

    args.add_argument('-test', '--test_data', default='data/test.txt',
                      help='Path to test data file')
    return args.parse_args()


def main(args):
    pp = Preprocessing()

    # load data
    print("Loading Data.....\n\n")
    train_block, train_block_label = pp.read_train_file(args.train_data, args.train_label)
    test_block = pp.read_test_file(args.test_data)

    # explore data, do some visualization
    print("Exploring Data (see 'fig' folder for visualization) .....\n\n")
    viz = Visualization()

    # histogram for the lpc coefficient distribution
    viz.visualize_lpc_distribution(train_block)

    # histogram for the block length (or point of time) distribution
    viz.visualize_block_length_distribution(train_block)

    # plot one block of lpc coefficient for each speaker to look at the pattern of voice frequency
    viz.visualize_lpc_time_series(train_block)
    viz.visualize_fitted_lpc_series(train_block)

    max_length = 29
    final_block_size = 18

    print("Data Preprocessing (padding to fixed size blocks)....\n\n")
    # Take the best lengths (18), truncate the longer block, and pad the  shorter block by the last row
    train_data = pp.pad_to_fixed_size_blocks(train_block, max_length, final_block_size)
    test_data = pp.pad_to_fixed_size_blocks(test_block, max_length, final_block_size)

    # dummy test label for convenience
    test_block_label = [[i] for i in np.zeros(len(test_data))]

    print("Generating Features (for ML Algorithms)... \n\n")

    # Generate fixed length feature vector for traditional machine learning input
    final_train_data = pp.convert_to_vectors(train_data, train_block_label, final_block_size)
    final_test_data = pp.convert_to_vectors(test_data, test_block_label, final_block_size)

    # See scatter plot to find out if there is grouping based on feature vector
    viz.lpc_scatter_plot(final_train_data)

    # Looks like there is a grouping, so let's try to classify using some popular algorithm
    model = Models()
    model.run_classification_models(final_train_data, final_test_data)

    print("SVM Prediction Saved (see 'results/submission.txt' )... \n\n")

    #Also try LSTM for classification
    model.run_LSTM_model(np.array(train_data), np.array(train_block_label), np.array(test_data))
    print("LSTM Prediction Saved (see 'results/submission_lstm.txt' )... \n\n")

if __name__=="__main__":
    args = parse_args()
    main(args)