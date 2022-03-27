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

FUNC(void, GLOBAL) processEdges(P2CONST(int, DEVICE) adjMatrix, P2VAR(int, DEVICE) shortestDistances, P2VAR(int, DEVICE) updateShortestDistances, P2VAR(int, DEVICE) processedVertices, P2CONST(int, DEVICE) numVertices) 
{

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < *numVertices) {


        if (processedVertices[threadId] == MARKED) {

            processedVertices[threadId] = NOT_MARKED;

            for (int edge = 0; edge < *numVertices; edge++) {
                if (adjMatrix[threadId * (*numVertices) + edge] != 0)
                {
                    atomicMin(&updateShortestDistances[edge], shortestDistances[threadId] + adjMatrix[threadId * (*numVertices) + edge]);
                }
            }
        }
    }
}

FUNC(void, GLOBAL) relaxEdges(P2CONST(int, DEVICE) adjMatrix, P2VAR(int, DEVICE) shortestDistances, P2VAR(int, DEVICE) updateShortestDistances, P2VAR(int, DEVICE) processedVertices, P2CONST(int, DEVICE) numVertices) 
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



FUNC(void, HOST) Dijkstra(P2CONST(int, HOST) adjMatrix, P2VAR(int, HOST) shortestDistances, P2VAR(int, HOST) updateShortestDistances, P2VAR(int, HOST) processedVertices, P2CONST(int, HOST) numVertices, CONSTVAR(int, HOST) testCaseNumber)
{

    /*Config timer data*/
    cudaEvent_t startTimerEvent, stopTimerEvent;
    float elapsedTimeMs = 0;
    CUDA_SAFE_CALL(cudaEventCreate(&startTimerEvent));
    CUDA_SAFE_CALL(cudaEventCreate(&stopTimerEvent));

    /*GPU device global variables*/
    int* gpuNumVertices;
    int* gpuAdjMatrix;
    int* gpuShortestDistances;
    int* gpuUpdateShortestDistances;
    int* gpuProcessedVertices;

    /*Init the gpu device input variables*/
    cudaMalloc((void**)&gpuNumVertices, (*numVertices) * sizeof(int));
    cudaMalloc((void**)&gpuAdjMatrix, (*numVertices) * (*numVertices) * sizeof(int));
    cudaMalloc((void**)&gpuShortestDistances, (*numVertices) * sizeof(int));
    cudaMalloc((void**)&gpuUpdateShortestDistances, (*numVertices) * sizeof(int));
    cudaMalloc((void**)&gpuProcessedVertices, (*numVertices) * sizeof(int));

    /*Copy the host data to gpu*/
    cudaMemcpy(gpuNumVertices, numVertices, (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuAdjMatrix, adjMatrix, (*numVertices) * (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuShortestDistances, shortestDistances, (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuProcessedVertices, processedVertices, (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuUpdateShortestDistances, updateShortestDistances, (*numVertices) * sizeof(int), cudaMemcpyHostToDevice);


    /*Run Dijkstra parallel algorithm*/
    CUDA_SAFE_CALL(cudaEventRecord(startTimerEvent));

    while (STD_NOT_OK == allVerticesProcessed(processedVertices, numVertices)) {

        for (int asyncIt = 0; asyncIt < NUM_ASYNCHRONOUS_ITERATIONS; asyncIt++) {

            processEdges << < (*numVertices) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (gpuAdjMatrix, gpuShortestDistances, gpuUpdateShortestDistances, gpuProcessedVertices, gpuNumVertices);

            //CUDA_SAFE_CALL(cudaPeekAtLastError());
            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            relaxEdges << < (*numVertices) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (gpuAdjMatrix, gpuShortestDistances, gpuUpdateShortestDistances, gpuProcessedVertices, gpuNumVertices);

            //CUDA_SAFE_CALL(cudaPeekAtLastError());
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
        }

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
    CUDA_SAFE_CALL(cudaFree(gpuAdjMatrix));
    CUDA_SAFE_CALL(cudaFree(gpuShortestDistances));
    CUDA_SAFE_CALL(cudaFree(gpuUpdateShortestDistances));
    CUDA_SAFE_CALL(cudaFree(gpuProcessedVertices));
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
