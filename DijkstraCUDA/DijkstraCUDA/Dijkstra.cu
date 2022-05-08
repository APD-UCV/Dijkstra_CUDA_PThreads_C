#include "Dijkstra.cuh"

char parrentDirectoryPath[MAX_BUF];
char inputTestsPath[MAX_BUF];
char outputTestsPath[MAX_BUF];

FUNC(void, HOST) getParentDirectoryPath()
{

    if (GetCurrentDirectory(MAX_BUF, parrentDirectoryPath) == NULL)
    {
        printf("[ERROR] the path to the tests directory fetch failed\n");
        exit(EXIT_FAILURE);
    }

    int counter = 0;
    for (int i = MAX_BUF; i >= 0; --i)
    {
        if (parrentDirectoryPath[i] == '\\')
        {
            counter += 1;
        }


        if (counter == 2) {
            parrentDirectoryPath[i] = '\0';
            break;
        }
    }
}

FUNC(P2VAR(Node, HOST), HOST) readInputData(P2VAR(int, HOST) numVertices, P2VAR(int, HOST) numEdges)
{
    int cudaGpuDeviceId = cudaGetDevice(&cudaGpuDeviceId);
    FILE* pf = fopen(inputTestsPath, "r");;

    if (pf == NULL)
    {
        printf("Invalid input file");
        return NULL;
    }

    fscanf(pf, "%d %d", numVertices, numEdges);

    Node* graph;

    cudaMallocManaged((void**)&graph, (*numVertices) * sizeof(Node));

    int no_neighbors = -1;
    for (int i = 0; i < *numVertices; i++)
    {
        fscanf(pf, "%d", &no_neighbors);
        graph[i].no_neighbors = no_neighbors;

        if (no_neighbors != 0)
        {
            cudaMalloc((void**)&graph[i].adj_list, no_neighbors * 2 * sizeof(int));
        }
    }

    int startEdge;
    int endEdge;
    int weight;
    for (int i = 0; i < *numVertices; i++)
    {
        if (graph[i].no_neighbors == 0)
            continue;

        int* temp = (int*)malloc(graph[i].no_neighbors * 2 * sizeof(int));

        for (int j = 0; j < 2 * graph[i].no_neighbors; j += 2)
        {
            fscanf(pf, "%d %d %d", &startEdge, &endEdge, &weight);
            temp[j] = endEdge;
            temp[j + 1] = weight;
        }

        cudaMemcpy(graph[i].adj_list, temp, graph[i].no_neighbors * 2 * sizeof(int), cudaMemcpyHostToDevice);
        free(temp);
    }

    fclose(pf);
    return graph;
}

FUNC(void, HOST) printCollectedData(P2CONST(int, HOST) shortestDistances, P2CONST(int, HOST) numVertices, CONSTVAR(float, HOST) elapsedTimeMs, CONSTVAR(int, HOST) testCaseNumber)
{
    /*Cumpute the path to the current test case*/
    outputTestsPath[strlen(outputTestsPath) - 5] = testCaseNumber + '0';
    FILE* pf = fopen(outputTestsPath, "w");

    fprintf(pf, "\n\nCUDA Time (ms): %7.9f\n", elapsedTimeMs);

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

FUNC(void, GLOBAL) processEdges(P2CONST(Node, DEVICE) graph, P2VAR(int, DEVICE) shortestDistances, P2VAR(int, DEVICE) updateShortestDistances, P2VAR(int, DEVICE) processedVertices, P2CONST(int, DEVICE) numVertices)
{

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < *numVertices) {

        if (processedVertices[threadId] == MARKED) {

            processedVertices[threadId] = NOT_MARKED;

            for (int i = 0; i < 2 * graph[threadId].no_neighbors; i+=2) {
                atomicMin(&updateShortestDistances[graph[threadId].adj_list[i]], shortestDistances[threadId] + graph[threadId].adj_list[i + 1]);
            }
        }
    }
}

FUNC(void, GLOBAL) relaxEdges(P2CONST(Node, DEVICE) graph, P2VAR(int, DEVICE) shortestDistances, P2VAR(int, DEVICE) updateShortestDistances, P2VAR(int, DEVICE) processedVertices, P2CONST(int, DEVICE) numVertices)
{

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < *numVertices) {

        if (shortestDistances[threadId] > updateShortestDistances[threadId]) {
            shortestDistances[threadId] = updateShortestDistances[threadId];
            processedVertices[threadId] = MARKED;
        }

        updateShortestDistances[threadId] = shortestDistances[threadId];
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

FUNC(void, HOST) Dijkstra(P2VAR(Node, HOST) graph, P2VAR(int, HOST) shortestDistances, P2VAR(int, HOST) updateShortestDistances,
    P2VAR(int, HOST) processedVertices, P2VAR(int, HOST) numVertices, CONSTVAR(int, HOST) testCaseNumber)
{

    /*Config timer data*/
    cudaEvent_t startTimerEvent, stopTimerEvent;
    float elapsedTimeMs = 0;
    CUDA_SAFE_CALL(cudaEventCreate(&startTimerEvent));
    CUDA_SAFE_CALL(cudaEventCreate(&stopTimerEvent));

    /*GPU device global variables*/
    int* gpuNumVertices;
    int* gpuShortestDistances;
    int* gpuUpdateShortestDistances;
    int* gpuProcessedVertices;

    /*Init the gpu device input variables*/
    cudaMalloc((void**)&gpuNumVertices, (*numVertices) * sizeof(int));
    cudaMalloc((void**)&gpuShortestDistances, (*numVertices) * sizeof(int));
    cudaMalloc((void**)&gpuUpdateShortestDistances, (*numVertices) * sizeof(int));
    cudaMalloc((void**)&gpuProcessedVertices, (*numVertices) * sizeof(int));

    /*Copy the host data to gpu*/
    cudaMemcpy(gpuNumVertices, numVertices, (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuShortestDistances, shortestDistances, (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuProcessedVertices, processedVertices, (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuUpdateShortestDistances, updateShortestDistances, (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);

    /*Run Dijkstra parallel algorithm*/
    CUDA_SAFE_CALL(cudaEventRecord(startTimerEvent));

    while (STD_NOT_OK == allVerticesProcessed(processedVertices, numVertices)) {

        processEdges << < (*numVertices) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (graph, gpuShortestDistances, gpuUpdateShortestDistances, gpuProcessedVertices, gpuNumVertices);

        cudaDeviceSynchronize();

        relaxEdges << < (*numVertices) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (graph, gpuShortestDistances, gpuUpdateShortestDistances, gpuProcessedVertices, gpuNumVertices);

        cudaDeviceSynchronize();
        
        CUDA_SAFE_CALL(cudaMemcpy(processedVertices, gpuProcessedVertices, sizeof(int) * (*numVertices), cudaMemcpyDeviceToHost));
    }

    CUDA_SAFE_CALL(cudaEventRecord(stopTimerEvent));
    cudaEventSynchronize(stopTimerEvent);

    /*Calculate elapsed time*/
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTimeMs, startTimerEvent, stopTimerEvent));


    /*Send the generated data to host*/
    CUDA_SAFE_CALL(cudaMemcpy(shortestDistances, gpuShortestDistances, sizeof(int) * (*numVertices), cudaMemcpyDeviceToHost));


    /*Print collected data*/
    printCollectedData(shortestDistances, numVertices, elapsedTimeMs, testCaseNumber);

    /*Free gpu device memory*/
    CUDA_SAFE_CALL(cudaFree(gpuNumVertices));
    CUDA_SAFE_CALL(cudaFree(gpuShortestDistances));
    CUDA_SAFE_CALL(cudaFree(gpuUpdateShortestDistances));
    CUDA_SAFE_CALL(cudaFree(gpuProcessedVertices));
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
                CUDA_SAFE_CALL(cudaFree(graph[i].adj_list));
        }

        free(numEdges);
        free(numVertices);
        free(updateShortestDistances);
        free(shortestDistances);
        free(processedVertices);
        CUDA_SAFE_CALL(cudaFree(graph));

    }
}
