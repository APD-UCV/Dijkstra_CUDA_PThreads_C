#include "Dijkstra.h"

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

FUNC(P2VAR(int, HOST), HOST) readInputData(P2VAR(int, HOST) numVertices)
{
    FILE* pf = fopen(inputTestsPath, "r");;

    if (pf == NULL)
    {
        printf("Invalid input file");
        return NULL;
    }

    fscanf(pf, "%d", numVertices);

    int* adjMatrix = (int*)malloc(*numVertices * *numVertices * sizeof(int));
    for (int i = 0; i < *numVertices; ++i)
    {
        for (int j = 0; j < (*numVertices); ++j)
        {
            fscanf(pf, "%d ", &adjMatrix[i * (*numVertices) + j]);
        }
    }

    fclose(pf);
    return adjMatrix;
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

FUNC(int, HOST) closestVertex(P2CONST(int, HOST) shortestDistances, P2CONST(int, HOST) processedVertices, P2CONST(int, HOST) numVertices)
{
    int min = INF_DIST;
    int min_index = -1;
    for(int vertexIt = 0; vertexIt < *numVertices; ++vertexIt)
    {
        if(processedVertices[vertexIt] == NOT_MARKED && shortestDistances[vertexIt] <= min)
        {
            min = shortestDistances[vertexIt];
            min_index = vertexIt;
        }
    }
    return min_index;
}

FUNC(void, HOST) Dijkstra(P2CONST(int, HOST) adjMatrix, P2VAR(int, HOST) shortestDistances, P2VAR(int, HOST) processedVertices, P2CONST(int, HOST) numVertices, CONSTVAR(int, HOST) testCaseNumber)
{
    /*Config timer data*/
    clock_t startTimerEvent, stopTimerEvent;
    float elapsedTimeMs = 0;
    
    /*Run Dijkstra parallel algorithm*/
    startTimerEvent = clock();

    for(int vertexIt = 0; vertexIt < *numVertices - 1; ++vertexIt)
    {
        int startVertex = closestVertex(shortestDistances, processedVertices, numVertices);
        processedVertices[startVertex] = MARKED;
        for(int endVertex = 0; endVertex <*numVertices; ++endVertex)
        {
            if(processedVertices[endVertex] == NOT_MARKED && adjMatrix[startVertex * (*numVertices) + endVertex] != 0 
                && shortestDistances[startVertex] + adjMatrix[startVertex * (*numVertices) + endVertex] < shortestDistances[endVertex])
            {
                shortestDistances[endVertex] = shortestDistances[startVertex] + adjMatrix[startVertex * (*numVertices) + endVertex];
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
    int* adjMatrix;
    int* shortestDistances;
    int* processedVertices;

    /*Run the algorithm for every test case*/
    for (int i = 0; i < 8; i++) {

        /*Cumpute the path to the current test case*/
        inputTestsPath[strlen(inputTestsPath) - 4] = i + '0';
        printf("Starting test %d\n", i);

        /*Init the host input variables*/

        numVertices = (int*)malloc(sizeof(int));
        adjMatrix = readInputData(numVertices);
        if (adjMatrix == NULL) {
            printf("\nInput fetch failed\n");
            return;
        }

        /*Init host global variables*/
        shortestDistances = (int*)malloc(*numVertices * sizeof(int));
        processedVertices = (int*)malloc(*numVertices * sizeof(int));

        initArray(shortestDistances, numVertices, INF_DIST);
        initArray(processedVertices, numVertices, (int)NOT_MARKED);

        shortestDistances[SOURCE_VERTEX] = 0;

        /*Run dijkstra*/
        Dijkstra(adjMatrix, shortestDistances, processedVertices, numVertices, i);

        printf("Test %d finished\n\n", i);

        /*Free host memory*/
        free(numVertices);
        free(adjMatrix);
        free(shortestDistances);
        free(processedVertices);


    }
}
