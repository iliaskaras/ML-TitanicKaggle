from Algorithms.SVM import SVM
from DAO.FileWriter import FileWriter

fileWriter = FileWriter()


'''' The C parameter tells the SVM_results optimization how much you want to avoid misclassifying each training example.
     For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better
     job of getting all the training points classified correctly. Conversely, a very small value of C will cause the
     optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.
     For very tiny values of C, you should get misclassified examples, often even if your training data is linearly separable. '''

'''========================================== LINEAR ============================================================'''

''' With big Penalty, and Linear '''

kernel = 'linear'
c = 100
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.readFile("../../outputFiles/SVM_results/SVM_Outputs.txt")
fileWriter.writeFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

''' With small Penalty, and Linear '''
kernel = 'linear'
c = 20
svmAlgorithm = None
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())


''' With really small Penalty, and Linear '''
kernel = 'linear'
c = 0.01
svmAlgorithm = None
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

'''========================================== POLY ============================================================'''

''' With big Penalty, and poly '''

kernel = 'poly'
c = 100
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

''' With small Penalty, and poly '''
kernel = 'poly'
c = 20
svmAlgorithm = None
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

''' With really small Penalty, and poly '''
kernel = 'poly'
c = 1
svmAlgorithm = None
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

svmAlgorithm.initPredictionFile()

submissionDf = svmAlgorithm.getFinalSubmissionDf()
fileWriter.readSubmissionFile("../../outputFiles/SVM_results/gender_submission.csv")
fileWriter.writeSubmissionFile(submissionDf)

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

'''========================================== RBF ============================================================'''

''' With big Penalty, and rbf '''

kernel = 'rbf'
c = 100
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

''' With small Penalty, and rbf '''
kernel = 'rbf'
c = 20
svmAlgorithm = None
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

''' With really small Penalty, and rbf '''
kernel = 'rbf'
c = 1
svmAlgorithm = None
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

'''========================================== SIGMOID ============================================================'''

''' With big Penalty, and sigmoid '''

kernel = 'sigmoid'
c = 100
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

''' With small Penalty, and sigmoid '''
kernel = 'sigmoid'
c = 20
svmAlgorithm = None
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())

''' With really small Penalty, and sigmoid '''
kernel = 'sigmoid'
c = 1
svmAlgorithm = None
svmAlgorithm = SVM()
svmAlgorithm.initClf(kernel, c)
svmAlgorithm.fitClf()
svmAlgorithm.initPredictions()
svmAlgorithm.getScores()

fileWriter.appendFile(svmAlgorithm.get_MethodDetailsMsg() + svmAlgorithm.get_AccuracyMsg() +
                     svmAlgorithm.get_PrecisionMsg() + svmAlgorithm.get_RecallMsg() + svmAlgorithm.get_F1Msg())
