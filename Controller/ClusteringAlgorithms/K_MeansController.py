
from Algorithms.K_Means import K_Means
from DAO.FileWriter import FileWriter


#ORIGINAL:
# For a clustering algorithm, the machine will find the clusters, but then will asign arbitrary values to them,
# in the order it finds them. Thus, the group that is survivors might be a 0 or a 1, depending on a degree of randomness.
# Thus, if you consistently get 30% and 70% accuracy, then your model is 70% accurate. Let's see what we get:



fileWriter = FileWriter()

kMeansClf = K_Means()
kMeansClf.createK_MeansModel(2, 'micro', init='k-means++')
kMeansClf.initPredictions()
kMeansClf.getValidations()

fileWriter.readFile("../../outputFiles/K_Means_results/K_Means_Outputs.txt")
fileWriter.writeFile(kMeansClf.get_MethodDetailsMsg() + kMeansClf.get_Adjusted_rand_scoreMsg() + kMeansClf.get_Silhouette_scoreMsg()
                     + kMeansClf.get_Accuracy_scoreMsg())

kMeansClf.initPredictionFile()

submissionDf = kMeansClf.getFinalSubmissionDf()
fileWriter.readSubmissionFile("../../outputFiles/K_Means_results/gender_submission.csv")
fileWriter.writeSubmissionFile(submissionDf)


kMeansClf = K_Means()
kMeansClf.createK_MeansModel(2, 'micro', init='random')
kMeansClf.getValidations()

fileWriter.appendFile(kMeansClf.get_MethodDetailsMsg() + kMeansClf.get_Adjusted_rand_scoreMsg() + kMeansClf.get_Silhouette_scoreMsg()
                      + kMeansClf.get_Accuracy_scoreMsg())



# try:
#     from pyspark import SparkContext
#     from pyspark import SparkConf
#     print ("Successfully imported Spark Modules")
# except ImportError as e:
#     print ("Can not import Spark Modules", e)
#
# sc = SparkContext('local')
# words = sc.parallelize(["scala","java","hadoop","spark","akka"])
# print(words.count())

