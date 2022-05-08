#ifndef DIJKSTRA_H
#define DIJKSTRA_H

/* Standard library imports */
#include <float.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <Windows.h>

/*CUDA specific imports*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* Local application / library specific imports */
#include "ProjectDefinitions.cuh"
#include "Node.cuh"
#include "MinHeap.cuh"

/*Config application specific macros*/
#define INF_DIST (10000000)                 // Initial infinite distance between two nodes
#define THREADS_PER_BLOCK (128)				// Number of threads assigned per block
#define MAX_BUF (200)						// Maximum size of path string
#define SOURCE_VERTEX (0)					// Source vertex

/*Input and output path variables*/
EXTERN char parrentDirectoryPath[MAX_BUF];
EXTERN char inputTestsPath[MAX_BUF];
EXTERN char outputTestsPath[MAX_BUF];

/*Input and output specific functions*/
EXTERN FUNC(void, HOST) getParentDirectoryPath();
EXTERN FUNC(P2VAR(int, HOST), HOST) readInputData(P2VAR(int, HOST) numVertices);
EXTERN  FUNC(void, HOST) printCollectedData(P2CONST(int, HOST) shortestDistances, P2CONST(int, HOST) numVertices, CONSTVAR(float, HOST) elapsedTimeMs, CONSTVAR(int, HOST) testCaseNumber);

/*Util functions*/
EXTERN  FUNC(void, HOST) initArray(P2VAR(int, HOST) arrayData, P2CONST(int, HOST) size, CONSTVAR(int, HOST) initValue);

/*CUDA PROCESSING FUNCTIONS*/
EXTERN  FUNC(void, GLOBAL) processEdges(P2CONST(Node, DEVICE) graph, P2VAR(int, DEVICE) shortestDistances, P2VAR(int, DEVICE) updateShortestDistances, 
	P2VAR(int, DEVICE) processedVertices, P2CONST(int, DEVICE) numVertices);
EXTERN  FUNC(void, GLOBAL) relaxEdges(P2CONST(Node, DEVICE) graph, P2VAR(int, DEVICE) shortestDistances, P2VAR(int, DEVICE) updateShortestDistances,
	P2VAR(int, DEVICE) processedVertices, P2CONST(int, DEVICE) numVertices);

EXTERN  FUNC(int, HOST) allVerticesProcessed(P2CONST(int, HOST) processedVertices, P2CONST(int, HOST) numVertices);
EXTERN  FUNC(void, HOST) Dijkstra(P2VAR(Node, HOST) graph, P2VAR(int, HOST) shortestDistances, P2VAR(int, HOST) updateShortestDistances,
							P2VAR(int, HOST) processedVertices, P2VAR(int, HOST) numVertices, CONSTVAR(int, HOST) testCaseNumber);

/*Start tests suite function*/
EXTERN FUNC(void, HOST) startTests();

#endif // !DIJKSTRA_H
