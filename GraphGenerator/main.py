import os
import random
import time

class GraphGenerator:

    def __init__(self) -> None:
        random.seed(time.time())

    def generateGraphSuite(self, noGraphs: int) -> None:
        noVertices = 128
        for graphID in range(noGraphs):

            test_name = "test" + str(graphID) + ".in"
            dir = "\\".join(os.getcwd().split("\\")[:-1])
            file_name = dir + "\\input_files\\{0}".format(test_name)

            out = open(file_name, "w")

            out.write(str(noVertices) + "\n")

            for vertexID in range(noVertices):
                neighborsPerVertex = 6 #random.randint(1, noVertices//20)
                edges = [0 for i in range(noVertices)]
                for neighborVertexID in range(neighborsPerVertex):
                    stopLoop = False
                    while not stopLoop:
                        stopLoop = True
                        destinationVertex = random.randint(0, noVertices - 1)
                        if edges[destinationVertex] != 0:
                            stopLoop = False
                        if destinationVertex is vertexID:
                            stopLoop = False
                        if stopLoop:
                            edges[destinationVertex] = random.randint(1, 40)
                            break
                for destinationVertex in range(noVertices):
                    out.write(str(edges[destinationVertex]) + " ")
                out.write("\n")

            print("Graph {0}".format(graphID) + "generated")
            out.close()
            noVertices *= 2


if __name__ == '__main__':
    graphGenerator = GraphGenerator()
    graphGenerator.generateGraphSuite(10)
