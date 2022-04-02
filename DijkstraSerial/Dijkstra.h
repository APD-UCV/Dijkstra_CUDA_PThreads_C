#ifndef DIJKSTRA_H
#define DIJKSTRA_H

/* Standard library imports */
#include <float.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

/*Unix specific imports*/
#include <unistd.h>

/* Local application / library specific imports */
#include "ProjectDefinitions.h"


/*Config application specific macros*/
#define NUM_ASYNCHRONOUS_ITERATIONS (20)    // Number of async loop iterations before attempting to read results back
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
STATIC FUNC(void, HOST) printCollectedData(P2CONST(int, HOST) shortestDistances, P2CONST(int, HOST) numVertices, CONSTVAR(float, HOST) elapsedTimeMs, CONSTVAR(int, HOST) testCaseNumber);


/*Util functions*/
STATIC FUNC(void, HOST) initArray(P2VAR(int, HOST) arrayData, P2CONST(int, HOST) size, CONSTVAR(int, HOST) initValue);

// /*Processing functions*/
STATIC FUNC(int, HOST) closestVertex(P2CONST(int, HOST) shortestDistances, P2CONST(int, HOST) processedVertices, P2CONST(int, HOST) numVertices);
STATIC FUNC(void, HOST) Dijkstra(P2CONST(int, HOST) adjMatrix, P2VAR(int, HOST) shortestDistances, P2VAR(int, HOST) processedVertices, P2CONST(int, HOST) numVertices, CONSTVAR(int, HOST) testCaseNumber);

// /*Start tests suite function*/
EXTERN FUNC(void, HOST) startTests();

#endif // !DIJKSTRA_H
