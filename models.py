# ******************************************************************************
# models.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/27/19   Paudel     Initial version,
# ******************************************************************************

from numpy import mean
from numpy import std
import pandas as pd

from matplotlib import pyplot

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import (Dense, LSTM, Dropout)

from exploration import Visualization

class Models():
    def __init__(self):
        pass

    # Function to print result of the classifier
    def print_result(self, y_test, y_pred):
        '''
        print classification result (on train set)
        :param y_test:
        :param y_pred:
        :return:
        '''
        print("\n\n")
        print(metrics.classification_report(y_test, y_pred))
        print("Confusion Matrix: \n\n", metrics.confusion_matrix(y_test, y_pred))

    def save_prediction(self, test_X, predict, file_name):
        '''
        save prediction to the txt file
        :param test_X:
        :param predict:
        :param file_name:
        :return:
        '''
        predictions = []
        for block in range(0, len(test_X)):
            predictions.append([block,int(predict[block])])
        submission = pd.DataFrame(predictions, columns=['block_num', 'prediction'])
        submission.to_csv(file_name, header=['block_num','prediction'], index=None, sep=',')

    def build_train_lstm_model(self, trainX, trainy):
        '''
        build lstm model and train
        :param trainX:
        :param trainy:
        :return:
        '''
        trainy = to_categorical(trainy)
        verbose, epochs, batch_size = 2, 150, 64
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save('lstm_model.h5')
        return history

    def lstm_predict(self, testX):
        '''
        load the saved model and predict the class for test data
        :param testX:
        :return:
        '''
        model = load_model('lstm_model.h5')
        return model.predict_classes(testX)

    def compare_ml_models(self, train_X, train_y, nfold):
        '''
        Initial experimentation on several algorithms
        :param train_X:
        :param train_y:
        :param nfold:
        :return:
        '''
        models, names = list(), list()
        # knn
        models.append(KNeighborsClassifier())
        names.append('KNN')
        # logistic
        models.append(LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200))
        names.append('LR')
        # cart
        models.append(DecisionTreeClassifier())
        names.append('CART')
        # svm
        models.append(SVC())
        names.append('SVM')
        # random forest
        models.append(RandomForestClassifier(n_estimators=100))
        names.append('RF')
        # evaluate models

        all_scores = list()
        for i in range(len(models)):
            s = StandardScaler()
            p = Pipeline(steps=[('s', s), ('m', models[i])])
            scores = cross_val_score(p, train_X, train_y, scoring='accuracy', cv=nfold, n_jobs=-1)
            all_scores.append(scores)
            m, s = mean(scores) * 100, std(scores) * 100
            print('%s %.3f%% +/-%.3f' % (names[i], m, s))

        return all_scores, names

    def svm_parameter_tuning(self, train_X, train_y, nfold):
        '''
        find best parameter for svm
        :param train_X:
        :param train_y:
        :param nfold:
        :return:
        '''
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel':['rbf','linear']}
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfold, verbose=1)
        grid_search.fit(train_X, train_y)
        grid_search.best_params_
        return grid_search.best_params_, grid_search.best_estimator_

    def test_best_model(self, best_estimator, train_X, train_y):
        '''

        :param best_estimator:
        :param train_X:
        :param train_y:
        :return:
        '''
        y_pred = cross_val_predict(best_estimator, train_X, train_y, cv=5)
        print("Performance by SVM (with best estimator) in test data using cross-val...\n\n")
        self.print_result(train_y, y_pred)

    def classify_test_data(self, best_estimator, train_X, train_y, test_X):
        '''
        classify test data using best svm settings
        :param best_estimator:
        :param train_X:
        :param train_y:
        :param test_X:
        :return:
        '''
        best_estimator.fit(train_X, train_y)
        predict = best_estimator.predict(test_X)
        return predict

    def run_classification_models(self, train_data, test_data):
        '''
        use traditional machine learning approach
        :param train_data:
        :param test_data:
        :return:
        '''
        train_X, train_y = train_data[:,:-1], train_data[:,-1]
        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        nfold = 5

        print("Running Algorithms for Spot Checking ... \n\n")
        all_scores, names = self.compare_ml_models(train_X, train_y, nfold)

        # # Visualize boxplot to see the best model
        pyplot.boxplot(all_scores, labels=names)
        # pyplot.show()
        pyplot.savefig('fig/spot_check_box_plot.png')

        # Since SVM shows the best performance.. Let's tune the parameter for SVM and find the best model
        print("Running Grid Search for the Best Algorithm (SVM) ... \n\n")
        best_param, best_estimator = self.svm_parameter_tuning(train_X, train_y, nfold)
        print("Best Parameters:      ", best_param)
        print("\n\nBest Estimators:      ", best_estimator)

        # test the performance of best svm model
        self.test_best_model(best_estimator, train_X, train_y)
        print("Predicting Speakers..... \n\n")
        #predict the speaker for test data using best svm model
        predict = self.classify_test_data(best_estimator, train_X, train_y, test_X)
        self.save_prediction(test_X, predict, 'results/submission.txt')

    def run_LSTM_model(self, trainX, trainy, testX):
        '''
        use lstm based classifer for speaker classification
        :param trainX:
        :param trainy:
        :param testX:
        :return:
        '''
        # build and train LSTM Model
        print("\n\nTraining LSTM....\n\n")
        history = self.build_train_lstm_model(trainX, trainy)

        # visualize the training error and convergence of LSTM
        viz = Visualization()
        viz.show_training_history(history)

        #predict the speaker using
        predict = self.lstm_predict(testX)
        self.save_prediction(testX, predict, 'results/submission_lstm.txt')
