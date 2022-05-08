import os
import random
import time

from networkx.generators.random_graphs import fast_gnp_random_graph


class GraphGenerator:

    def generateGraphSuite(self, noGraphs: int) -> None:
        noVertices = 1024
        probability = 0.001
        for graphID in range(noGraphs):

            test_name = "test" + str(graphID) + ".in"
            dir = "\\".join(os.getcwd().split("\\")[:-1])
            file_name = dir + "\\input_files\\{0}".format(test_name)

            GRAPH = fast_gnp_random_graph(noVertices, probability, random.seed(time.time()), True)

            out = open(file_name, "w")
            out.write(str(noVertices) + " " + str(len(GRAPH.edges)) + "\n")
            for i in range(noVertices):
                out.write(str(len(GRAPH.edges(i))) + " ")
            out.write("\n")
            for edge in GRAPH.edges:
                out.write(str(edge[0]) + " " + str(edge[1]) + " " + str(random.randint(1, 100)) + "\n")

            print("Graph {0}".format(graphID) + "generated")
            out.close()
            noVertices *= 2


if __name__ == '__main__':
    graphGenerator = GraphGenerator()
    graphGenerator.generateGraphSuite(8)
