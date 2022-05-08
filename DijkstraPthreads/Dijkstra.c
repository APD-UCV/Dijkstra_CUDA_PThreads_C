#include "Dijkstra.h"

char parrentDirectoryPath[MAX_BUF];
char inputTestsPath[MAX_BUF];
char outputTestsPath[MAX_BUF];
threadpool thpool;

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

    Node* graph = (Node*)malloc((*numVertices) * sizeof(Node)); 

    int no_neighbors = -1;
    for(int i =0; i < *numVertices; i++)
    {
        fscanf(pf, "%d", &no_neighbors);
        graph[i].no_neighbors = no_neighbors;

        if (no_neighbors != 0)
        {
            graph[i].adj_list = (int*)malloc(no_neighbors * 2 * sizeof(int));
        }
    }

    int startEdge;
    int endEdge;
    int weight;
    for(int i =0; i < *numVertices; i++)
    {
        if(graph[i].no_neighbors == 0)
            continue;
        for(int j =0; j < 2 * graph[i].no_neighbors; j += 2)
        {
            fscanf(pf, "%d %d %d", &startEdge, &endEdge, &weight);
            graph[i].adj_list[j] = endEdge;
            graph[i].adj_list[j+1] = weight;
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

void* processEdges(P2VAR(void, HOST) args)
{
    GraphInfo_t* graphInfo = (GraphInfo_t*)args;
    int threadId = graphInfo->threadId;

    if (threadId < *(graphInfo->numVertices)) {

        if (graphInfo->processedVertices[threadId] == MARKED) {

            graphInfo->processedVertices[threadId] = NOT_MARKED;

            for(int i=0; i < 2 * graphInfo->graph[threadId].no_neighbors; i+=2)
            {
                if (graphInfo->shortestDistances[threadId] + graphInfo->graph[threadId].adj_list[i+1] < graphInfo->updateShortestDistances[graphInfo->graph[threadId].adj_list[i]])
                {
                    pthread_mutex_lock(graphInfo->mutex_lock);
                    graphInfo->updateShortestDistances[graphInfo->graph[threadId].adj_list[i]] = graphInfo->shortestDistances[threadId] + graphInfo->graph[threadId].adj_list[i+1];
                    pthread_mutex_unlock(graphInfo->mutex_lock);
                }
            }   
        }
    }
}

void* relaxEdges(P2VAR(void, HOST) args)
{

    GraphInfo_t* graphInfo = (GraphInfo_t*)args;
    int threadId = graphInfo->threadId;

    if (threadId < *(graphInfo->numVertices)) {
        if (graphInfo->shortestDistances[threadId] > graphInfo->updateShortestDistances[threadId]) {
            graphInfo->shortestDistances[threadId] = graphInfo->updateShortestDistances[threadId];
            graphInfo->processedVertices[threadId] = MARKED;
        }

        graphInfo->updateShortestDistances[threadId] = graphInfo->shortestDistances[threadId];
    }
}


FUNC(void, HOST) Dijkstra(P2VAR(Node, HOST) graph, P2VAR(int, HOST) shortestDistances, P2VAR(int, HOST) updateShortestDistances, P2VAR(int, HOST) processedVertices, P2VAR(int, HOST) numVertices, CONSTVAR(int, HOST) testCaseNumber)
{
    /*Config timer data*/
    clock_t startTimerEvent, stopTimerEvent;
    float elapsedTimeMs = 0;


    GraphInfo_t* graphInfo;
    pthread_mutex_t* mutex_lock;

    mutex_lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(mutex_lock, NULL);

    graphInfo = (GraphInfo_t*)malloc(*numVertices * sizeof(GraphInfo_t));
    for(int threadId =0; threadId < *numVertices; ++threadId)
    {
        graphInfo[threadId].graph = graph;
        graphInfo[threadId].shortestDistances = shortestDistances;
        graphInfo[threadId].updateShortestDistances = updateShortestDistances;
        graphInfo[threadId].processedVertices = processedVertices;
        graphInfo[threadId].numVertices = numVertices;
        graphInfo[threadId].threadId = threadId;
        graphInfo[threadId].mutex_lock = mutex_lock;        
    }
    
    /*Run Dijkstra parallel algorithm*/
    startTimerEvent = clock();

    while (STD_NOT_OK == allVerticesProcessed(processedVertices, numVertices)) {
            
        for (int threadId = 0; threadId < *numVertices; ++threadId) {
            thpool_add_work(thpool, (void*)processEdges, (void*) &graphInfo[threadId]);
        }
        thpool_wait(thpool);
        for (int threadId = 0; threadId < *numVertices; ++threadId) {
            thpool_add_work(thpool, (void*)relaxEdges, (void*) &graphInfo[threadId]);                
        }
        thpool_wait(thpool);
        
    }

    stopTimerEvent = clock();

    /*Calculate elapsed time*/
    elapsedTimeMs = ((float)(stopTimerEvent - startTimerEvent)/CLOCKS_PER_SEC) * 1000;

    /*Print collected data*/
    printCollectedData(shortestDistances, numVertices, elapsedTimeMs, testCaseNumber);

    /*Free threads used memory*/
    //thpool_destroy(thpool);
    pthread_mutex_destroy(mutex_lock);
    free(graphInfo);

}

FUNC(void, HOST) startTests()
{

    /*Host global variables*/
    int* numVertices;
    int* numEdges;
    Node* graph;
    int* shortestDistances;
    int* updateShortestDistances;
    int* processedVertices;
    thpool = thpool_init(THREAD_POOL_SIZE);

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
        updateShortestDistances = (int*)malloc(*numVertices * sizeof(int));
        processedVertices = (int*)malloc(*numVertices * sizeof(int));

        initArray(shortestDistances, numVertices, INF_DIST);
        initArray(updateShortestDistances, numVertices, INF_DIST);
        initArray(processedVertices, numVertices, (int)NOT_MARKED);

        /*Init host global variables*/
        shortestDistances[SOURCE_VERTEX] = 0;
        updateShortestDistances[SOURCE_VERTEX] = 0;
        processedVertices[SOURCE_VERTEX] = MARKED;

        /*Run dijkstra*/
        Dijkstra(graph, shortestDistances, updateShortestDistances, processedVertices, numVertices, i);

        printf("Test %d finished\n\n", i);

        /*Free host memory*/

        for (int i = 0; i < *numVertices; i++)
        {
            if(graph[i].no_neighbors != 0)
                free(graph[i].adj_list);
        }


        free(numEdges);
        free(numVertices);
        free(updateShortestDistances);
        free(shortestDistances);
        free(processedVertices);
        free(graph);
    }


    thpool_destroy(thpool);

}