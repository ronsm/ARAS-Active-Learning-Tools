import csv
import numpy as np
import pandas as pd
import sys
import pickle
from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier 
from skmultiflow.bayes import NaiveBayes
from sklearn.metrics import confusion_matrix, classification_report

from log import Log

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

class ARASActiveLearningTools(object):
    def __init__(self):
        self.id = 'ARAS_active_learning_tools'
        
        self.logger = Log(self.id)

        self.load_data()
        self.load_models()

    def load_data(self):
        self.logger.log('Loading data files: train, validation, and annotations...')

        self.train = pd.read_csv('data/train.csv', dtype=int)
        print(self.train.shape)
        print(self.train.head())

        self.header = list(self.train.columns.values)

        self.validation = pd.read_csv('data/validation.csv', dtype=int)
        print(self.validation.shape)
        print(self.validation.head())

        self.annotations = pd.read_csv('data/annotations.csv', dtype=int)
        self.annotations.columns = self.header
        print(self.annotations.shape)
        print(self.annotations.head())

    def load_merged(self):
        self.merged = pd.read_csv('data/merged.csv', dtype=int)
        print(self.merged.shape)
        print(self.merged.head())

    def load_models(self):
        self.model_1_train = pickle.load(open('models/train/Model1.p', 'rb'))
        self.model_2_train = pickle.load(open('models/train/Model2.p', 'rb'))
        self.model_3_train = pickle.load(open('models/train/Model3.p', 'rb'))

        self.model_1_with_annotations = pickle.load(open('models/with_annotations/Model1.p', 'rb'))
        self.model_2_with_annotations = pickle.load(open('models/with_annotations/Model2.p', 'rb'))
        self.model_3_with_annotations = pickle.load(open('models/with_annotations/Model3.p', 'rb'))

    def merge(self):
        self.logger.log('Merging the annotated data with the original training set...')

        self.merged = pd.concat([self.train, self.annotations])
        print(self.merged.shape)
        print(self.merged.head())

        self.logger.log('Saving dataframe...')

        self.merged.to_csv('data/merged.csv', index=False)

        self.logger.log_great('Done.')

    # see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics for available metrics
    def validate(self):
        self.load_merged()
        y_true = self.validation['R1'].tolist()

        models_train = [self.model_1_train, self.model_2_train, self.model_3_train]
        models_with_annotations = [self.model_1_with_annotations, self.model_2_with_annotations, self.model_3_with_annotations]

        self.logger.log('Predicting with models trained only on the training set...')

        for model in models_train:
            self.stream = FileStream('data/validation.csv')
            y_pred_train = self.predict(model, self.stream)
            cm = classification_report(y_true, y_pred_train, labels=labels)
            print(cm)

        self.logger.log('Predicting with models trained with the annotations added...')

        for model in models_with_annotations:
            self.stream = FileStream('data/validation.csv')
            y_pred_train = self.predict(model, self.stream)
            cm = classification_report(y_true, y_pred_train, labels=labels)
            print(cm)

    def predict(self, model, stream):
        y_pred = []
        count = 0
        while stream.has_more_samples():
            X, y = stream.next_sample()
            y_pred.append(model.predict(X))
            if (count % 50000) == 0:
                print('Predictions so far:', count)
            count = count + 1

        return y_pred

if __name__ == '__main__':
    aras_alt = ARASActiveLearningTools()

    mode = sys.argv[1]

    if mode == 'merge':
        aras_alt.merge()
    elif mode == 'validate':
        aras_alt.validate()
    else:
        print('Invalid mode specified. Valid modes: merge, validate')