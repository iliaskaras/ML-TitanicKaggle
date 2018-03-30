import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.cross_validation import cross_val_score

from Controller.DataHandler.DataHandler import DataHandler


class SVM(object):

    def __init__(self):
        self.nonNumericalDataHandler = DataHandler()
        self.loadTrainDF()
        self.loadTestDF()
        self.loadCompareDF()

    """ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' """
    """ Penalty  C : float, optional (default=1.0)
        Penalty parameter C of the error term. """

    def initClf(self, kernel, penalty):
        self.kernel = kernel
        self.penalty = str(penalty)
        self.clf = svm.SVC(kernel=kernel, C=penalty)
        print(
            " ============================= kernel : " + self.kernel + " =====penalty : " + self.penalty + " =====================\n")

    def fitClf(self):
        if self.clf is not None:
            self.clf.fit(self.X_train, self.y_train)
            print(" ============================= Clf Fit Complete =================================\n")
        else:
            print("                          (Warning)                               \n"
                  "You must first create the Classifier and then try to get the fit !\n")

    def loadTrainDF(self):
        self.dfTrain = pd.read_csv("../../resources/train.csv")
        self.textTrainPreprocess()
        self.initXtrain()

    def loadTestDF(self):
        self.dfTest = pd.read_csv("../../resources/test.csv")
        self.textTestPreprocess()
        self.initXY_test()

    def loadCompareDF(self):
        self.dfCompareTest = pd.read_csv("../../resources/gender_submission.csv", usecols=[1])
        self.dfComparePassengerIds = pd.read_csv("../../resources/gender_submission.csv", usecols=[0])
        self.textComparePreprocess()
        self.initY_compare()

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

    def textComparePreprocess(self):
        self.dfCompareTest.fillna(0, inplace=True)
        self.dfCompareTest = self.nonNumericalDataHandler.changeNonNumericalData(self.dfCompareTest)

    def initXtrain(self):
        self.X = np.array(self.dfTrain.drop(['Survived'], 1).astype(float))
        self.X = preprocessing.scale(self.X)
        self.y = np.array(self.dfTrain['Survived'])
        xLength = self.X.__len__()
        trainN = round(xLength * 0.7)
        testN = round(xLength * 0.3)
        self.X_train, self.X_test = self.X[:trainN], self.X[-testN:]
        self.y_train, self.y_test = self.y[:trainN], self.y[-testN:]

    def initXY_test(self):
        self.XtestUnlabeled = np.array(self.dfTest.astype(float))
        self.XtestUnlabeled = preprocessing.scale(self.XtestUnlabeled)

    def initY_compare(self):
        self.yTest = np.array(self.dfCompareTest['Survived'])
        self.yTestPassengerId = np.array(self.dfComparePassengerIds['PassengerId'])

    def initPredictions(self):
        self.YtestUnlabeledPredictions = self.getPrediction()

    def getPrediction(self):
        global yTestPredictions
        if self.clf is not None:
            yTestPredictions = self.clf.predict(self.XtestUnlabeled)
            # print(yTestPredictions)
            # print(" ===================== Predictions Generated Successfully =======================\n")
        else:
            print("                          (Warning)                               \n"
                  "You must first create the Classifier and then try to get the predictions !\n")
        return yTestPredictions

    def initPredictionFile(self):
        self.createDataframe(self.YtestUnlabeledPredictions)

    def createDataframe(self, YtestUnlabeledPredictions):
        yTestPredictions_dataframe = pd.DataFrame(YtestUnlabeledPredictions, columns=['Survived'])
        yTestPassengerId_dataframe = pd.DataFrame(self.yTestPassengerId, columns=['PassengerId'])
        self.finalSubmissionDf = yTestPassengerId_dataframe.join(yTestPredictions_dataframe)

    def getFinalSubmissionDf(self):
        return self.finalSubmissionDf

    def getScores(self):
        self.getPrecisionScore()
        self.getRecallScore()
        self.getF1Score()
        self.getAccuracyScore()
        return None

    def getAccuracyScore(self):
        if self.clf is not None:
            crossValResults = cross_val_score(self.clf, self.X_test, self.y_test, scoring="accuracy")

            average = np.mean(crossValResults)
            self.accuracy = " Average Accuracy score of the 5 training sets : " + str(average) + "\n"

            print(" Average Accuracy score of the 5 training sets : " + str(average))
            print("\n")

        else:
            self.warningMessage("precision")

    def getPrecisionScore(self):
        if self.clf is not None:
            crossValResults = cross_val_score(self.clf, self.X_test, self.y_test, cv=5, scoring="precision")
            average = np.mean(crossValResults)
            self.precision = " Average Precision score of the 5 training sets : " + str(average) + "\n"

            print(" Average Precision score of the 5 training sets : " + str(average))
            print("\n")

        else:
            self.warningMessage("precision")

    def getRecallScore(self):
        if self.clf is not None:
            crossValResults = cross_val_score(self.clf, self.X_test, self.y_test, cv=5, scoring="recall")
            average = np.mean(crossValResults)
            self.recall = " Average Recall score of the 5 training sets : " + str(average) + "\n"
            print(" Average Recall score of the 5 training sets : " + str(average))
            print("\n")

        else:
            self.warningMessage("recall")

    def getF1Score(self):
        if self.clf is not None:
            crossValResults = cross_val_score(self.clf, self.X_test, self.y_test, cv=5, scoring="f1")
            average = np.mean(crossValResults)
            self.f1 = " Average F1 score of the 5 training sets : " + str(average) + "\n"

            print(" Average F1 score of the 5 training sets : " + str(average))
            print("\n")

        else:
            self.warningMessage("f1")

    def get_Xtrain(self):
        return self.Xtrain

    def get_Ytrain(self):
        return self.Ytrain

    def get_AccuracyMsg(self):
        return self.accuracy

    def get_F1Msg(self):
        return self.f1

    def get_RecallMsg(self):
        return self.recall

    def get_PrecisionMsg(self):
        return self.precision

    def get_MethodDetailsMsg(self):
        return " =================================================================================\n" \
               " SVM_results with Kernel : " + self.kernel + ", Penalty : " + self.penalty + "\n" \
               " =================================================================================\n"

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
