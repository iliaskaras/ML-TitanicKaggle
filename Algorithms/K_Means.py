import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, adjusted_rand_score, completeness_score, homogeneity_score, \
    silhouette_score
from sklearn.model_selection import cross_val_score
from pyspark import SparkContext
# always inherit from object in 2.x. it's called new-style classes.
from Controller.DataHandler.DataHandler import DataHandler

import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

class K_Means(object):

    def __init__(self):

        self.kMeans = None
        self.nonNumericalDataHandler = DataHandler()
        self.loadTrainDF()
        self.loadTestDF()
        self.loadCompareDF()


    def createK_MeansModel(self, n_clusters, average, init):
        self.average = average
        self.n_clusters = n_clusters
        self.init = init
        self.kMeans = KMeans(n_clusters=n_clusters, init=self.init, random_state=0)

        # self.kMeans.fit(self.X_train, self.y_train)
        self.kMeans.fit(self.X)

        print(" ========= n_clusters: " + str(n_clusters)+" ========= ")

    def getValidations(self):

        if self.kMeans is not None:

            self.getScores()

        else:

            print("                          (Warning)                                    \n"
                  "You must first create the Classifier and then get the cross vailidation!\n"
                  "You can do that by simple call createDecisionTreeClassifier method.\n")

    def getScores(self):

        correct = 0
        for i in range(len(self.X)):

            predict_me = np.array(self.X[i].astype(float))
            predict_me = predict_me.reshape(-1, len(predict_me))
            prediction = self.kMeans.predict(predict_me)
            if prediction == self.y[i]:
                correct += 1

        print(correct / len(self.X))
        self.accuracy_score = " Accuracy score : " + str(correct / len(self.X)) + "\n"
        # correct = 0
        # for i in range(len(self.X_test)):
        #
        #     predict_me = np.array(self.X_test[i].astype(float))
        #     predict_me = predict_me.reshape(-1, len(predict_me))
        #     prediction = self.kMeans.predict(predict_me)
        #     if prediction == self.y_test[i]:
        #         correct += 1
        #
        # print(correct / len(self.X_test))
        # self.accuracy_score = " Accuracy score : " + str(correct / len(self.X_test)) + "\n"

        self.getAdjusted_rand_score()
        self.getVMeasure_score()
        self.getHomogeneity_score()
        self.getCompleteness_score()
        self.getSilhouette_score()

        return None

    def getSilhouette_score(self):
        if self.kMeans is not None:

            cluster_labels = self.kMeans.fit_predict(self.X)
            silhouetteScore = silhouette_score(self.X, cluster_labels)
            self.silhouetteScore = " Silhouette score : " + str(silhouetteScore) + "\n"

            print(" Silhouette score : " + str(silhouetteScore))
            print("\n")

        else:

            self.warningMessage("getSilhouette_score error")

    def getAdjusted_rand_score(self):
        if self.kMeans is not None:

            # y_pred = self.kMeans.predict(self.X_test)
            # adjustedRandScore = adjusted_rand_score(self.y_test, y_pred)
            y_pred = self.kMeans.predict(self.X)
            adjustedRandScore = adjusted_rand_score(self.y, y_pred)
            self.adjusted_rand_score = " Adjusted Rand score : " + str(adjustedRandScore) + "\n"

            print(" Adjusted Rand score : " + str(adjustedRandScore))
            print("\n")

        else:

            self.warningMessage("getAdjusted_rand_score error")


    def getVMeasure_score(self):
        if self.kMeans is not None:

            y_pred = self.kMeans.predict(self.X)
            vMeasureScore = v_measure_score(self.y, y_pred)
            self.vMeasureScore = " V Measure score : " + str(vMeasureScore) + "\n"

            print(" V Measure score : " + str(vMeasureScore))
            print("\n")

        else:

            self.warningMessage("getVMeasure_score error")

    def getCompleteness_score(self):
        if self.kMeans is not None:

            y_pred = self.kMeans.predict(self.X)
            completenessScore = completeness_score(self.y, y_pred)
            self.completenessScore = " Completeness score : " + str(completenessScore) + "\n"

            print(" Completeness score : " + str(completenessScore))
            print("\n")

        else:

            self.warningMessage("getCompleteness_score error")

    def getHomogeneity_score(self):
        if self.kMeans is not None:

            y_pred = self.kMeans.predict(self.X)
            homogeneityScore = homogeneity_score(self.y, y_pred)
            self.homogeneityScore = " Homogeneity score : " + str(homogeneityScore) + "\n"

            print(" Homogeneity score : " + str(homogeneityScore))
            print("\n")

        else:

            self.warningMessage("getHomogeneity_score error")

    def keepGoodResults(self, crossValResults):

        result = crossValResults[crossValResults >= 0.5]

        return result


    def get_featureNames(self):

        self.dfTrain.drop(['Survived'], 1, inplace=True)

        return self.dfTrain.columns.values.tolist()


    def get_targetNames(self):

        return self.dfTest.columns.values.tolist()

    def get_clf(self):

        return self.kMeans

    def warningMessage(self, typeOfMessage):

        if typeOfMessage == "f1":
            print("                          (Warning)                                    \n"
                  "You must first create the Classifier and then get the f1 score!\n"
                  "You can do that by simple call createDecisionTreeClassifier method.\n")

        elif typeOfMessage == "precision":
            print("                          (Warning)                                    \n"
                  "You must first create the Classifier and then get the precision score!\n"
                  "You can do that by simple call createDecisionTreeClassifier method.\n")

        elif typeOfMessage == "recall":
            print("                          (Warning)                                    \n"
                  "You must first create the Classifier and then get the recall score!\n"
                  "You can do that by simple call createDecisionTreeClassifier method.\n")

    def loadTrainDF(self):

        self.dfTrain = pd.read_csv("../../resources/train.csv")
        self.textTrainPreprocess()
        self.initXtrain()


    def loadTestDF(self):

        self.dfTest = pd.read_csv("../../resources/test.csv")
        self.textTestPreprocess()
        self.initX_test()


    def textTrainPreprocess(self):
        self.dfTrain.drop(['Name'], 1, inplace=True)
        self.dfTrain.drop(['PassengerId'], 1, inplace=True)
        self.dfTrain.drop(['Ticket'], 1, inplace=True)
        self.dfTrain.drop(['Cabin'], 1, inplace=True)
        dfList = self.dfTrain['Age'].tolist()
        grps = np.arange(0, 100, 30)
        dfList = np.digitize(dfList, grps)
        self.dfTrain['Age'] = dfList

        self.dfTrain = self.nonNumericalDataHandler.changeNonNumericalData(self.dfTrain)
        self.dfTrain.fillna(0, inplace=True)

    def textTestPreprocess(self):

        self.dfTest.drop(['Name'], 1, inplace=True)
        self.dfTest.drop(['PassengerId'], 1, inplace=True)
        self.dfTest.drop(['Ticket'], 1, inplace=True)
        self.dfTest.drop(['Cabin'], 1, inplace=True)
        dfList = self.dfTest['Age'].tolist()
        grps = np.arange(0, 100, 30)
        dfList = np.digitize(dfList, grps)
        self.dfTest['Age'] = dfList

        self.dfTest = self.nonNumericalDataHandler.changeNonNumericalData(self.dfTest)
        self.dfTest.fillna(0, inplace=True)

    def initXtrain(self):
        self.X = np.array(self.dfTrain.drop(['Survived'], 1).astype(float))
        # self.X = np.array(self.dfTrain).astype(float)

        self.X = preprocessing.scale(self.X)
        self.y = np.array(self.dfTrain['Survived'])

        # xLength = self.X.__len__()
        # trainN = round(xLength * 0.7)
        # testN = round(xLength * 0.3)
        # self.X_train, self.X_test = self.X[:trainN], self.X[-testN:]
        # self.y_train, self.y_test = self.y[:trainN], self.y[-testN:]


    def get_Accuracy_scoreMsg(self):

        return self.accuracy_score

    def get_Adjusted_rand_scoreMsg(self):

        return self.adjusted_rand_score

    def get_Silhouette_scoreMsg(self):

        return self.silhouetteScore

    def get_VMeasure_Msg(self):

        return self.vMeasureScore

    def get_Homogeneity_Msg(self):

        return self.homogeneityScore

    def get_Completeness_Msg(self):

        return self.completenessScore

    def get_MethodDetailsMsg(self):
        return " =================================================================================\n" \
               " ====================== K_Means with  Clusters : " + str(self.n_clusters) + " ============\n" \
               " =================================================================================\n"

    def initPredictions(self):
        self.YtestUnlabeledPredictions = self.getPrediction()

    def getPrediction(self):
        global yTestPredictions
        if self.kMeans is not None:
            yTestPredictions = self.kMeans.predict(self.XtestUnlabeled)
        else:
            print("                          (Warning)                               \n"
                  "You must first create the Classifier and then try to get the predictions !\n")
        return yTestPredictions

    def initPredictionFile(self):
        self.createDataframe(self.YtestUnlabeledPredictions)

    def createDataframe(self,YtestUnlabeledPredictions):
        yTestPredictions_dataframe = pd.DataFrame(YtestUnlabeledPredictions, columns=['Survived'])
        yTestPassengerId_dataframe = pd.DataFrame(self.yTestPassengerId, columns=['PassengerId'])
        self.finalSubmissionDf = yTestPassengerId_dataframe.join(yTestPredictions_dataframe)

    def getFinalSubmissionDf(self):
        return self.finalSubmissionDf

    def initX_test(self):
        self.XtestUnlabeled = np.array(self.dfTest.astype(float))
        self.XtestUnlabeled = preprocessing.scale(self.XtestUnlabeled)


    def loadCompareDF(self):
        self.dfCompareTest = pd.read_csv("../../resources/gender_submission.csv", usecols=[1])
        self.dfComparePassengerIds = pd.read_csv("../../resources/gender_submission.csv", usecols=[0])
        self.textComparePreprocess()
        self.initY_compare()

    def textComparePreprocess(self):
        self.dfCompareTest.fillna(0, inplace=True)
        self.dfCompareTest = self.nonNumericalDataHandler.changeNonNumericalData(self.dfCompareTest)

    def initY_compare(self):
        self.yTest = np.array(self.dfCompareTest['Survived'])
        self.yTestPassengerId = np.array(self.dfComparePassengerIds['PassengerId'])