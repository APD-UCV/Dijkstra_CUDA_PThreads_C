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

    fprintf(pf, "\n\n Pthreads Time (ms): %7.9f\n", elapsedTimeMs);

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

FUNC(void, HOST) init2DArray(P2VAR(int, HOST) arrayData, P2CONST(int, HOST) size, CONSTVAR(int, HOST) initValue)
{

    for (int i = 0; i < *size; ++i)
    {
        for (int j = 0; j < *size; ++j)
        {
            arrayData[i * (*size) + j] = initValue;
        }
    }
}

FUNC(P2VAR(void, HOST), HOST) processEdges(P2VAR(void, HOST) args)
{
    GraphInfo_t* graphInfo = (GraphInfo_t*)args;
    int threadId = *(graphInfo->threadId);

    if (threadId < *(graphInfo->numVertices)) {

        if (graphInfo->processedVertices[threadId] == MARKED) {

            graphInfo->processedVertices[threadId] = NOT_MARKED;

            for (int edge = 0; edge < *(graphInfo->numVertices); edge++) {
                if (graphInfo->adjMatrix[threadId * *(graphInfo->numVertices) + edge] != 0)
                {
                    pthread_mutex_lock(graphInfo->mutex_lock);

                    graphInfo->updateShortestDistances[edge] = MIN(graphInfo->updateShortestDistances[edge], 
                            graphInfo->shortestDistances[threadId] + graphInfo->adjMatrix[threadId * *(graphInfo->numVertices) + edge]);

                    pthread_mutex_unlock(graphInfo->mutex_lock);
                }
            }
        }
    }

    pthread_exit(NULL);

}

FUNC(P2VAR(void, HOST), HOST) relaxEdges(P2VAR(void, HOST) args)
{

    GraphInfo_t* graphInfo = (GraphInfo_t*)args;
    int threadId = *(graphInfo->threadId);

    if (threadId < *(graphInfo->numVertices)) {
        if (graphInfo->shortestDistances[threadId] > graphInfo->updateShortestDistances[threadId]) {
            graphInfo->shortestDistances[threadId] = graphInfo->updateShortestDistances[threadId];
            graphInfo->processedVertices[threadId] = MARKED;
        }

        graphInfo->updateShortestDistances[threadId] = graphInfo->shortestDistances[threadId];
    }

    pthread_exit(NULL);
}

FUNC(STD_RETURN_TYPE, HOST) allVerticesProcessed(P2CONST(int, HOST) processedVertices, P2CONST(int, HOST) numVertices)
{

    for (int i = 0; i < *numVertices; i++)
    {
        if (processedVertices[i] == MARKED)
        {
            return STD_NOT_OK;
        }
    }

    return STD_OK;
}

FUNC(void, HOST) synchronizePthreads(P2CONST(pthread_t, HOST) threadList, P2CONST(int, HOST) listSize)
{
    for(int threadId=0; threadId < *listSize; ++threadId){
	    pthread_join(threadList[threadId], NULL);		
	}
}

FUNC(void, HOST) Dijkstra(P2VAR(int, HOST) adjMatrix, P2VAR(int, HOST) shortestDistances, P2VAR(int, HOST) updateShortestDistances, P2VAR(int, HOST) processedVertices, P2VAR(int, HOST) numVertices, CONSTVAR(int, HOST) testCaseNumber)
{
    /*Config timer data*/
    clock_t startTimerEvent, stopTimerEvent;
    float elapsedTimeMs = 0;

    pthread_t* threadList;
    pthread_attr_t attr;
    GraphInfo_t* graphInfo;
    pthread_mutex_t* mutex_lock;

    mutex_lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(mutex_lock, NULL);

    graphInfo = (GraphInfo_t*)malloc(*numVertices * sizeof(GraphInfo_t));
    for(int threadId =0; threadId < *numVertices; ++threadId)
    {
        graphInfo[threadId].adjMatrix = adjMatrix;
        graphInfo[threadId].shortestDistances = shortestDistances;
        graphInfo[threadId].updateShortestDistances = updateShortestDistances;
        graphInfo[threadId].processedVertices = processedVertices;
        graphInfo[threadId].numVertices = numVertices;
        graphInfo[threadId].threadId = (int*)malloc(sizeof(int));
        *(graphInfo[threadId].threadId) = threadId;
        graphInfo[threadId].mutex_lock = mutex_lock;        
    }
    
    threadList = (pthread_t*)malloc(*numVertices * sizeof(pthread_t));

    pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    /*Run Dijkstra parallel algorithm*/
    startTimerEvent = clock();

    while (STD_NOT_OK == allVerticesProcessed(processedVertices, numVertices)) {

        for (int asyncIt = 0; asyncIt < NUM_ASYNCHRONOUS_ITERATIONS; ++asyncIt) {
            
            for (int threadId = 0; threadId < *numVertices; ++threadId) {
                pthread_create(&threadList[threadId], &attr, (void*)processEdges, (void*) &graphInfo[threadId]);         
            }

	        synchronizePthreads(threadList, numVertices);

            for (int threadId = 0; threadId < *numVertices; ++threadId) {
                pthread_create(&threadList[threadId], &attr, (void*)relaxEdges, (void*) &graphInfo[threadId]);         
            }

	        synchronizePthreads(threadList, numVertices);
        }
    }

    stopTimerEvent = clock();

    /*Calculate elapsed time*/
    elapsedTimeMs = (float)(stopTimerEvent - startTimerEvent)/CLOCKS_PER_SEC;

    /*Print collected data*/
    printCollectedData(shortestDistances, numVertices, elapsedTimeMs, testCaseNumber);

    /*Free threads used memory*/
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(mutex_lock);
	free(threadList);		
    for(int i =0; i<*numVertices; ++i)
    {
        free(graphInfo[i].threadId); 
    }
    free(graphInfo);

}

FUNC(void, HOST) startTests()
{

    /*Host global variables*/
    int* numVertices;
    int* adjMatrix;
    int* shortestDistances;
    int* updateShortestDistances;
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
        updateShortestDistances = (int*)malloc(*numVertices * sizeof(int));
        processedVertices = (int*)malloc(*numVertices * sizeof(int));


        initArray(shortestDistances, numVertices, INF_DIST);
        initArray(updateShortestDistances, numVertices, INF_DIST);
        initArray(processedVertices, numVertices, (int)NOT_MARKED);
        
        shortestDistances[SOURCE_VERTEX] = 0;
        updateShortestDistances[SOURCE_VERTEX] = 0;
        processedVertices[SOURCE_VERTEX] = MARKED;

        /*Run dijkstra*/
        Dijkstra(adjMatrix, shortestDistances, updateShortestDistances, processedVertices, numVertices, i);

        printf("Test %d finished\n\n", i);

        /*Free host memory*/
        free(numVertices);
        free(adjMatrix);
        free(shortestDistances);
        free(updateShortestDistances);
        free(processedVertices);

    }
}