from Algorithms.DecisionTreeClf import DecisionTreeClf
from DAO.FileWriter import FileWriter
from Graphiz.GraphvizCreator import GraphvizCreator


fileWriter = FileWriter()

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("gini", 50)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.readFile("../../outputFiles/DecisionTreeClf_results/DecisionTreeClf_outputs.txt")
fileWriter.writeFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                     decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())

graphvizCreator = GraphvizCreator(decisionTreeClf)
graphvizCreator.exportGraph()

decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("entropy", 50)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())


decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("gini", 20)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())


decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("entropy", 20)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())


decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("gini", 10)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())


decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("entropy", 10)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())



decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("gini", 5)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())


decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("entropy", 5)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

''' We create csv cause Entropy and Depth 5 gives the best accuracy '''
decisionTreeClf.initPredictionFile()

submissionDf = decisionTreeClf.getFinalSubmissionDf()
fileWriter.readSubmissionFile("../../outputFiles/DecisionTreeClf_results/gender_submission.csv")
fileWriter.writeSubmissionFile(submissionDf)

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())



decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("gini", 2)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())


graphvizCreator = GraphvizCreator(decisionTreeClf)
graphvizCreator.exportGraph()

decisionTreeClf = None

decisionTreeClf = DecisionTreeClf()
decisionTreeClf.createDecisionTreeClassifier("entropy", 2)
decisionTreeClf.initPredictions()
decisionTreeClf.getCrossValidation()

fileWriter.appendFile(decisionTreeClf.get_MethodDetailsMsg() + decisionTreeClf.get_AccuracyMsg() +
                      decisionTreeClf.get_PrecisionMsg() + decisionTreeClf.get_RecallMsg() + decisionTreeClf.get_F1Msg())