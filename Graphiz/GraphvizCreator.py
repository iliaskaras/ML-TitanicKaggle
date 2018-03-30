from sklearn import tree
import graphviz


class GraphvizCreator(object):

    def __init__(self,decisionTreeClf):
        self.initVariables(decisionTreeClf)

    def changeDecisionTreeClfObject(self, decisionTreeClf):
        self.initVariables(decisionTreeClf)
        print("new name : "+self.decisionTreeClf.get_name())

    def initVariables(self,decisionTreeClf):
        self.decisionTreeClf = decisionTreeClf
        self.clf = self.decisionTreeClf.get_clf()
        self.feature_names = self.decisionTreeClf.get_featureNames()
        self.class_names = self.decisionTreeClf.get_targetNames()
        self.outputFileName = "../../outputFiles/DecisionTree/"+self.decisionTreeClf.get_name() + "_" + self.decisionTreeClf.get_splitFunction() \
                              + "_" + str(self.decisionTreeClf.get_maxDepth()) + ".dot"

        self.initFittedClf()

    def initFittedClf(self):
        self.clf = self.clf.fit(self.decisionTreeClf.X_train,
                                self.decisionTreeClf.y_train)

    def exportGraph(self):
        dot_data = tree.export_graphviz(self.clf, out_file=None,
                                 feature_names= self.feature_names,
                                 class_names= self.class_names,
                                 filled=True, rounded=True,
                                 special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(filename=self.outputFileName)
