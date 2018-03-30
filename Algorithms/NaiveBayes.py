from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# always inherit from object in 2.x. it's called new-style classes.
from Controller.DataHandler.DataHandler import DataHandler


class NaiveBayes(object):
    def __init__(self):
        self.gausNB = None
        self.nonNumericalDataHandler = DataHandler()
        self.loadTrainDF()
        self.loadTestDF()
        self.loadCompareDF()

    def createAndFitGaussianNB(self, cv):
        self.cv = cv
        self.gausNB = GaussianNB()

        self.gausNB.fit(self.X_train, self.y_train)

        print(" ========= GaussianNB fit ended ========= ")

    def getValidations(self):
        if self.gausNB is not None:
            crossValResults = cross_val_score(self.gausNB, self.X_test, self.y_test, cv=self.cv )
            average = np.average(crossValResults)
            self.getScores()
            print(" Final Average of the 5 training sets : " + str(average))
            print("\n")

        else:
            print("                          (Warning)                                    \n"
                  "You must first create the Classifier and then get the cross vailidation!\n"
                  "You can do that by simple call createDecisionTreeClassifier method.\n")

    def getScores(self):
        self.getPrecisionScore()
        self.getRecallScore()
        self.getF1Score()
        self.getAccuracyScore()
        return None

    def getPrecisionScore(self):
        if self.gausNB is not None:
            crossValResults = cross_val_score(self.gausNB, self.X_test, self.y_test, cv=self.cv,
                                              scoring="precision")
            average = np.average(crossValResults)

            self.precision = " Average Precision score of the 5 training sets : " + str(average) + "\n"
            print(" Average Precision score of the 5 training sets : " + str(average))
            print("\n")

        else:
            self.warningMessage("precision")

    def getRecallScore(self):
        if self.gausNB is not None:
            crossValResults = cross_val_score(self.gausNB,  self.X_test, self.y_test, cv=self.cv,
                                              scoring="recall")
            average = np.average(crossValResults)

            self.recall = " Average Recall score of the 5 training sets : " + str(average) + "\n"

            print(" Average Recall score of the 5 training sets : " + str(average))
            print("\n")

        else:
            self.warningMessage("recall")

    def getF1Score(self):
        if self.gausNB is not None:
            crossValResults = cross_val_score(self.gausNB, self.X_test, self.y_test, cv=self.cv, scoring="f1")
            average = np.average(crossValResults)
            self.f1 = " Average F1 score of the 5 training sets : " + str(average) + "\n"

            print(" Average F1 score of the 5 training sets : " + str(average))
            print("\n")

        else:
            self.warningMessage("f1")

    def getAccuracyScore(self):
        if self.gausNB is not None:
            crossValResults = cross_val_score(self.gausNB, self.X_test, self.y_test, cv=self.cv,
                                              scoring="accuracy")
            average = np.average(crossValResults)
            self.accuracy = " Average Accuracy score of the 5 training sets : " + str(average) + "\n"

            print(" Average Accuracy score of the 5 training sets : " + str(average))

            print("\n")

        else:
            self.warningMessage("f1")

    def get_featureNames(self):
        return self.breastCancer.feature_names

    def get_targetNames(self):
        return self.breastCancer.target_names

    def get_clf(self):
        return self.gausNB

    def get_name(self):
        return self.name

    def get_maxDepth(self):
        return self.max_depth

    def get_splitFunction(self):
        return self.split_function

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
        if self.gausNB is not None:
            yTestPredictions = self.gausNB.predict(self.XtestUnlabeled)
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
               " ================================== Naive Bayes ==================================\n" \
               " =================================================================================\n"
