
#ifndef MINHEAP_H
#define MINHEAP_H

#include <stdio.h>
#include "ProjectDefinitions.h"

#define INF_DIST (10000000)                 // Initial infinite distance between two nodes

EXTERN long long TREE_ARRAY_SIZE;
EXTERN long long HEAP_SIZE;

typedef struct NodeStruct {
    int nodeId;
    int distance;
} Distance2Node;

/*Swap variables util function*/
FUNC(void, HOST) swap(P2VAR(Distance2Node, HOST) firstValue, P2VAR(Distance2Node, HOST) secondValue);
FUNC(int, HOST) getRightChild(int nodeIndex);
FUNC(int, HOST) getLeftChild(int nodeIndex);
FUNC(int, HOST) getParrent(int nodeIndex);
FUNC(void, HOST)  heapify(P2VAR(Distance2Node, HOST) minHeapArray, int index);
FUNC(void, HOST) decreaseHeapKey(P2VAR(Distance2Node, HOST) minHeapArray, int nodeIndex, Distance2Node key);
FUNC(void, HOST) insert(P2VAR(Distance2Node, HOST) minHeapArray, Distance2Node key);
FUNC(Distance2Node, HOST) extractMin(P2VAR(Distance2Node, HOST) minHeapArray);
FUNC(Distance2Node, HOST) getMin(P2VAR(Distance2Node, HOST) minHeapArray);

#endif // !DIJKSTRA_H
