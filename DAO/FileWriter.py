
class FileWriter(object):

    # The class "constructor" - It's actually an initializer
    def __init__(self):
        print()


    def readFile(self, fileName):
        self.file_object_write = open(fileName, "w")
        self.file_object_append = open(fileName, "a")


    def writeFile(self, resultsString):
        self.file_object_write.write(resultsString)

    def appendFile(self, resultsString):
        self.file_object_append.write("\n\n"+resultsString)


    def readSubmissionFile(self, fileName):
        self.fileName = fileName
        self.file_submission_write = open(fileName, "w")
        self.file_submission_append = open(fileName, "a")

    def writeSubmissionFile(self, dataframeToWrite):
        dataframeToWrite.to_csv(self.fileName, index=False)

    def appendSubmissionFile(self, resultsString):
        self.file_submission_append.write(resultsString)
