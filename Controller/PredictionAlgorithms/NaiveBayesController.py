from Algorithms.NaiveBayes import NaiveBayes
from DAO.FileWriter import FileWriter

fileWriter = FileWriter()

naiveBayesClf = NaiveBayes()
naiveBayesClf.createAndFitGaussianNB(cv=5)
naiveBayesClf.initPredictions()
naiveBayesClf.getValidations()

fileWriter.readFile("../../outputFiles/NaiveBayes_results/NaiveBayes_Outputs.txt")
fileWriter.writeFile(naiveBayesClf.get_MethodDetailsMsg() + naiveBayesClf.get_AccuracyMsg() +
                     naiveBayesClf.get_PrecisionMsg() + naiveBayesClf.get_RecallMsg() + naiveBayesClf.get_F1Msg())

naiveBayesClf.initPredictionFile()

submissionDf = naiveBayesClf.getFinalSubmissionDf()
fileWriter.readSubmissionFile("../../outputFiles/NaiveBayes_results/gender_submission.csv")
fileWriter.writeSubmissionFile(submissionDf)
