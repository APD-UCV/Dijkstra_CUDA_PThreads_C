#include "Dijkstra.h"
#include "MinHeap.h"
#include "Node.h"

long long TREE_ARRAY_SIZE = 0;
long long HEAP_SIZE = 0;

char parrentDirectoryPath[MAX_BUF];
char inputTestsPath[MAX_BUF];
char outputTestsPath[MAX_BUF];

FUNC(void, HOST) getParentDirectoryPath()
{

    if (getcwd(parrentDirectoryPath, MAX_BUF * sizeof(char)) == NULL)
    {
        printf("[ERROR] the path to the tests directory fetch failed\n");
        exit(EXIT_FAILURE);
    }

    int counter = 0;
    for (int i = MAX_BUF; i >= 0; --i)
    {
        if (parrentDirectoryPath[i] == '/')
        {
            counter += 1;
        }


        if (counter == 1) {
            parrentDirectoryPath[i] = '\0';
            break;
        }
    }
}

FUNC(P2VAR(Node, HOST), HOST) readInputData(P2VAR(int, HOST) numVertices, P2VAR(int, HOST) numEdges)
{
    FILE* pf = fopen(inputTestsPath, "r");;

    if (pf == NULL)
    {
        printf("Invalid input file");
        return NULL;
    }

    fscanf(pf, "%d %d", numVertices, numEdges);

    Node* graph = (Node*)malloc(*numVertices * sizeof(Node)); 

    int no_neighbors = -1;
    for(int i =0; i < *numVertices; i++)
    {
        fscanf(pf, "%d", &no_neighbors);
        graph[i].node_id = i;
        graph[i].no_neighbors = no_neighbors;
        if(no_neighbors == 0)
        {
            graph[i].adj_list = (int**)malloc(1 * sizeof(int*));
        }
        else
        {
            graph[i].adj_list = (int**)malloc(no_neighbors * sizeof(int*));
        }
        
        for(int j =0; j < no_neighbors; j++)
        {
            graph[i].adj_list[j] = (int*)malloc(2*sizeof(int));
        }
    }

    int startEdge;
    int endEdge;
    int weight;
    for(int i =0; i<*numVertices; i++)
    {
        if(graph[i].no_neighbors == 0)
            continue;
        for(int j =0; j<graph[i].no_neighbors; j++)
        {
            fscanf(pf, "%d %d %d", &startEdge, &endEdge, &weight);
            graph[i].adj_list[j][0] = endEdge;
            graph[i].adj_list[j][1] = weight;
        }
    }



    fclose(pf);
    return graph;
}

FUNC(void, HOST) printCollectedData(P2CONST(int, HOST) shortestDistances, P2CONST(int, HOST) numVertices, CONSTVAR(float, HOST) elapsedTimeMs, CONSTVAR(int, HOST) testCaseNumber)
{
    /*Cumpute the path to the current test case*/
    outputTestsPath[strlen(outputTestsPath) - 5] = testCaseNumber + '0';
    FILE* pf = fopen(outputTestsPath, "w");

    fprintf(pf, "\n\n Serial Time (ms): %7.9f\n", elapsedTimeMs);

    fprintf(pf, "\n\n");
    for (int i = 0; i < *numVertices; ++i)
    {
        fprintf(pf, "%d <-> %d     -> %d\n", SOURCE_VERTEX, i, shortestDistances[i]);
    }

    fclose(pf);

}

FUNC(void, HOST) initArray(P2VAR(int, HOST) arrayData, P2CONST(int, HOST) size, CONSTVAR(int, HOST) initValue)
{
    for (int i = 0; i < *size; ++i)
    {
        arrayData[i] = initValue;
    }
}

FUNC(void, HOST) Dijkstra(P2CONST(Node, HOST) graph, P2VAR(int, HOST) shortestDistances, P2CONST(int, HOST) numVertices, CONSTVAR(int, HOST) testCaseNumber)
{
    TREE_ARRAY_SIZE = *numVertices * 5;
    Distance2Node minHeap[TREE_ARRAY_SIZE];
    
    Distance2Node distance2Source = {SOURCE_VERTEX, 0};
    insert(minHeap, distance2Source);

    /*Config timer data*/
    clock_t startTimerEvent, stopTimerEvent;
    float elapsedTimeMs = 0;
    
    /*Run Dijkstra parallel algorithm*/
    startTimerEvent = clock();

    while(HEAP_SIZE > 0)
    {
        Distance2Node closestVertex = extractMin(minHeap); // closestVertex(shortestDistances, processedVertices, numVertices);

        if(closestVertex.distance > shortestDistances[closestVertex.nodeId])
        {
            continue;
        }

        for(int i=0; i<graph[closestVertex.nodeId].no_neighbors; i++)
        {
            if (closestVertex.distance + graph[closestVertex.nodeId].adj_list[i][1] < shortestDistances[graph[closestVertex.nodeId].adj_list[i][0]])
            {
                shortestDistances[graph[closestVertex.nodeId].adj_list[i][0]] = closestVertex.distance + graph[closestVertex.nodeId].adj_list[i][1];
                Distance2Node distance2EndVertex = {graph[closestVertex.nodeId].adj_list[i][0], shortestDistances[graph[closestVertex.nodeId].adj_list[i][0]]};
                insert(minHeap, distance2EndVertex);
            }
        }
    }

    stopTimerEvent = clock();

    /*Calculate elapsed time*/
    elapsedTimeMs = ((float)(stopTimerEvent - startTimerEvent)/CLOCKS_PER_SEC) * 1000;

    /*Print collected data*/
    printCollectedData(shortestDistances, numVertices, elapsedTimeMs, testCaseNumber);

}

FUNC(void, HOST) startTests()
{

    /*Host global variables*/
    int* numVertices;
    int* numEdges;
    Node* graph;
    int* shortestDistances;

    /*Run the algorithm for every test case*/
    for (int i = 0; i < 8; i++) {

        /*Cumpute the path to the current test case*/
        inputTestsPath[strlen(inputTestsPath) - 4] = i + '0';
        printf("Starting test %d\n", i);

        /*Init the host input variables*/

        numVertices = (int*)malloc(sizeof(int));
        numEdges = (int*)malloc(sizeof(int));

        graph = readInputData(numVertices, numEdges);

        /*Init host global variables*/
        shortestDistances = (int*)malloc(*numVertices * sizeof(int));

        initArray(shortestDistances, numVertices, INF_DIST);

        shortestDistances[SOURCE_VERTEX] = 0;


        // for(int i=0; i<*numVertices; i++)
        // {
        //     for(int j =0; j <graph[i].no_neighbors; j++)
        //     {
        //         printf("%d -> %d , %d\n", i, graph[i].adj_list[j][0], graph[i].adj_list[j][1]);
        //     }
        // }

        /*Run dijkstra*/
        Dijkstra(graph, shortestDistances, numVertices, i);

        printf("Test %d finished\n\n", i);

        /*Free host memory*/
        free(numVertices);
        free(shortestDistances);
        free(graph);

    }
}
