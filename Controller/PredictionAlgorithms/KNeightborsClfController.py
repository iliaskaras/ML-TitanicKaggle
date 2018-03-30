from Algorithms.KNeighborsClf import KNeighborsClf
from DAO.FileWriter import FileWriter

fileWriter = FileWriter()

''' ======================================== manhattan =============================================================='''

numberOfNeighbors = 5
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('manhattan', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.readFile("../../outputFiles/KNeighbors_results/KNeightbors_Outputs.txt")
fileWriter.writeFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                     kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


numberOfNeighbors = 10
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('manhattan', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


numberOfNeighbors = 15
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('manhattan', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

numberOfNeighbors = 40
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('manhattan', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

''' ======================================== minkowski =============================================================='''

numberOfNeighbors = 5
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('minkowski', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

numberOfNeighbors = 10
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('minkowski', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


numberOfNeighbors = 15
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('minkowski', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

numberOfNeighbors = 40
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('minkowski', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


''' ======================================== euclidean =============================================================='''

numberOfNeighbors = 5
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('euclidean', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())



numberOfNeighbors = 10
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('euclidean', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

kNeighborsClf.initPredictionFile()

submissionDf = kNeighborsClf.getFinalSubmissionDf()
fileWriter.readSubmissionFile("../../outputFiles/KNeighbors_results/gender_submission.csv")
fileWriter.writeSubmissionFile(submissionDf)

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


numberOfNeighbors = 15
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('euclidean', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

numberOfNeighbors = 40
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('euclidean', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


''' ======================================== chebyshev =============================================================='''

numberOfNeighbors = 5
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('chebyshev', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())



numberOfNeighbors = 10
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('chebyshev', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


numberOfNeighbors = 15
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('chebyshev', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

numberOfNeighbors = 40
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('chebyshev', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


numberOfNeighbors = 150
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('manhattan', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

numberOfNeighbors = 150
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('minkowski', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

numberOfNeighbors = 150
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('euclidean', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())

numberOfNeighbors = 150
kNeighborsClf = KNeighborsClf()
kNeighborsClf.createKNeighborsClf('chebyshev', 'distance', 'auto', numberOfNeighbors)
kNeighborsClf.initPredictions()
kNeighborsClf.getValidations()

fileWriter.appendFile(kNeighborsClf.get_MethodDetailsMsg() + kNeighborsClf.get_AccuracyMsg() +
                      kNeighborsClf.get_PrecisionMsg() + kNeighborsClf.get_RecallMsg() + kNeighborsClf.get_F1Msg())


