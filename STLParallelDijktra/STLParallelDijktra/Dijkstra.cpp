
/* Standard library imports */
#include "Dijkstra.h"
#include <filesystem>
#include <fstream>
#include <atomic>

std::string parrentDirectoryPath;
std::string inputTestsPath;
std::string outputTestsPath;

using std::filesystem::current_path;
std::atomic_flag lock = ATOMIC_FLAG_INIT;

void getParentDirectoryPath()
{
    parrentDirectoryPath = current_path().string();

    int counter = 0;
    for (int i = parrentDirectoryPath.size(); i >= 0; --i)
    {
        if (parrentDirectoryPath[i] == '\\')
        {
            counter += 1;
        }


        if (counter == 2) {
            parrentDirectoryPath = parrentDirectoryPath.substr(0, i);
            break;
        }
    }

    std::cout << parrentDirectoryPath<< std::endl;
}

void readInputData(std::vector<std::vector<std::pair<int, int>>>& graph, int& numVertices, int& numEdges)
{
    std::fstream in_file;
    in_file.open(inputTestsPath, std::ios::in);


    in_file >> numVertices >> numEdges;

    // START dummy reading -> needed to corespond with test cases made for c version and cuda
    int no_neighbors = -1;
    for (int i = 0; i < numVertices; i++)
    {
        std::vector<std::pair<int, int>> temp;
        graph.push_back(temp);
        in_file >> no_neighbors;
    }
    // END: dummy reading

    int startEdge;
    int endEdge;
    int weight;
    for (int i = 0; i < numEdges; i++)
    {
        in_file >> startEdge >> endEdge >> weight;
        graph[startEdge].push_back(std::make_pair(endEdge, weight));
    }

    in_file.close();

}

void printCollectedData(const std::vector<int>& shortestDistances, int numVertices, float elapsedTimeMs, int testCaseNumber)
{

    /*Cumpute the path to the current test case*/
    std::fstream out_file;
    outputTestsPath[outputTestsPath.size() - 5] = testCaseNumber + '0';
    out_file.open(outputTestsPath, std::ios::out | std::ios::trunc);
    
    out_file << std::setprecision(7) << std::fixed;

    out_file << "\n\n Serial Time (ms): " << elapsedTimeMs << "\n";

    out_file << "\n\n";
    for (int i = 0; i < numVertices; ++i)
    {
        out_file << SOURCE_VERTEX << " <-> " << i << "-> " << shortestDistances[i] << "\n";
    }

    out_file.close();
}

void initArray(std::vector<int>& arrayData, int size, int initValue)
{
    for (int i = 0; i < size; ++i)
    {
        arrayData.push_back(initValue);
    }
}

void Dijkstra(const std::vector<std::vector<std::pair<int, int>>>&graph, std::vector<int>& shortestDistances, int numVertices, int testCaseNumber)
{

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> Q;
    Q.push(std::make_pair(0, SOURCE_VERTEX));

    /*Config timer data*/
    clock_t startTimerEvent, stopTimerEvent;
    float elapsedTimeMs = 0;

    startTimerEvent = clock();

    while (!Q.empty())
    {
        auto src = Q.top();
        Q.pop();

        if (src.first > shortestDistances[src.second])
            continue;

        auto adj = graph[src.second];

#if(STD_ON == SERIAL)
        for (auto i : adj)
        {
            if (src.first + i.second < shortestDistances[i.first])
            {
                shortestDistances[i.first] = src.first + i.second;
                Q.push(std::make_pair(shortestDistances[i.first], i.first));
            }
        }
#else
        std::for_each(
            std::execution::par,
            adj.begin(),
            adj.end(),
            [&](auto&& item)
            {
                if (src.first + item.second < shortestDistances[item.first])
                {
                    shortestDistances[item.first] = src.first + item.second;

                    while (lock.test_and_set(std::memory_order_acquire)) {  // acquire lock
#if defined(__cpp_lib_atomic_flag_test)
                        while (lock.test(std::memory_order_relaxed))        // test lock
#endif
                            ; // spin
                    }

                    Q.push(std::make_pair(shortestDistances[item.first], item.first));

                    lock.clear(std::memory_order_release);                  // release lock
                }
            });
#endif 
    }

    stopTimerEvent = clock();

    /*Calculate elapsed time*/
    elapsedTimeMs = ((float)(stopTimerEvent - startTimerEvent) / CLOCKS_PER_SEC) * 1000;

    printCollectedData(shortestDistances, numVertices, elapsedTimeMs, testCaseNumber);
}

void startTests()
{

    /*Run the algorithm for every test case*/
    for (int i = 0; i < 8; i++) {

        /*Cumpute the path to the current test case*/
        inputTestsPath[inputTestsPath.size() - 4] = i + '0';
        std::cout << "Starting test " << i << std::endl;

        /*Init the host input variables*/
        int numVertices;
        int numEdges;
        std::vector<std::vector<std::pair<int, int>>> graph;
        std::vector<int> shortestDistances;

        readInputData(graph, numVertices, numEdges);

        initArray(shortestDistances, numVertices, INF_DIST);

        shortestDistances[SOURCE_VERTEX] = 0;

        /*Run dijkstra*/
        Dijkstra(graph, shortestDistances, numVertices, i);

        std::cout << "Test " << i << " finished\n";

    }
}
