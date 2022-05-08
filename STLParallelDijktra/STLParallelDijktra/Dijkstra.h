#ifndef DIJKSTRA_H
#define DIJKSTRA_H

/* Standard library imports */
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <semaphore>
#include <thread>
#include <execution>
#include <string>

/*Config application specific macros*/
#define STD_ON 1
#define STD_OFF 0

#define MAX_BUF (200)						// Maximum size of path string
#define SOURCE_VERTEX (0)					// Source vertex
#define INF_DIST (10000000)                 // Initial infinite distance between two nodes
#define SERIAL STD_OFF
/*Input and output path variables*/
extern std::string parrentDirectoryPath;
extern std::string inputTestsPath;
extern std::string outputTestsPath;


/*Input and output specific functions*/
void getParentDirectoryPath();
void readInputData(std::vector<std::vector<std::pair<int, int>>>& graph, int& numVertices, int& numEdges);
void printCollectedData(const std::vector<int>& shortestDistances, int numVertices, float elapsedTimeMs, int testCaseNumber);
void initArray(std::vector<int>& arrayData, int size, int initValue);


void Dijkstra(const std::vector<std::vector<std::pair<int, int>>>& graph, std::vector<int>& shortestDistances, int numVertices, int testCaseNumber);

// /*Start tests suite function*/
void startTests();

#endif // !DIJKSTRA_H
