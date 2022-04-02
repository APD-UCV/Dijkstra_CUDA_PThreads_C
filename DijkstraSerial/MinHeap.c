#include "MinHeap.h"

/*Swap variables util function*/
FUNC(void, HOST) swap(P2VAR(Distance2Node, HOST) firstValue, P2VAR(Distance2Node, HOST) secondValue) {
    Distance2Node temp;
    temp = (*firstValue);
    (*firstValue) = (*secondValue);
    (*secondValue) = temp;
}

/*Get the right child of the given index node*/
FUNC(int, HOST) getRightChild(int nodeIndex) {
  if((((2*nodeIndex)+1) < TREE_ARRAY_SIZE) && (nodeIndex >= 1))
    return (2*nodeIndex)+1;
  return -1;
}

/*Get the left child of the given index node*/
FUNC(int, HOST) getLeftChild(int nodeIndex) {
    if(((2*nodeIndex) < TREE_ARRAY_SIZE) && (nodeIndex >= 1))
        return 2*nodeIndex;
    return -1;
}

/*Get the parrent of the given index node*/
FUNC(int, HOST) getParrent(int nodeIndex) {
  if ((nodeIndex > 1) && (nodeIndex < TREE_ARRAY_SIZE)) {
    return nodeIndex/2;
  }
  return -1;
}

FUNC(void, HOST)  heapify(P2VAR(Distance2Node, HOST) minHeapArray, int index) {
  int leftChildIndex = getLeftChild(index);
  int rightChildIndex = getRightChild(index);

  // finding smallest among index, left child and right child
  int smallest = index;

  if ((leftChildIndex <= HEAP_SIZE) && (leftChildIndex>0)) {
    if (minHeapArray[leftChildIndex].distance < minHeapArray[smallest].distance) {
      smallest = leftChildIndex;
    }
  }

  if ((rightChildIndex <= HEAP_SIZE && (rightChildIndex>0))) {
    if (minHeapArray[rightChildIndex].distance < minHeapArray[smallest].distance) {
      smallest = rightChildIndex;
    }
  }

  // smallest is not the node, node is not a heap
  if (smallest != index) {
    swap(&minHeapArray[index], &minHeapArray[smallest]);
    heapify(minHeapArray, smallest);
  }
}

/*Decrease the index position of a key*/
void decreaseHeapKey(P2VAR(Distance2Node, HOST) minHeapArray, int nodeIndex, Distance2Node key) {
  minHeapArray[nodeIndex] = key;
  while((nodeIndex>1) && (minHeapArray[getParrent(nodeIndex)].distance > minHeapArray[nodeIndex].distance)) {
    swap(&minHeapArray[nodeIndex], &minHeapArray[getParrent(nodeIndex)]);
    nodeIndex = getParrent(nodeIndex);
  }
}

/*Insert a value in the min heap*/
void insert(P2VAR(Distance2Node, HOST) minHeapArray, Distance2Node key) {
  HEAP_SIZE++;
  minHeapArray[HEAP_SIZE].distance = INF_DIST;
  decreaseHeapKey(minHeapArray, HEAP_SIZE, key);
}

/*Extract the head of the tree i.e the min key*/
Distance2Node extractMin(P2VAR(Distance2Node, HOST) minHeapArray) {
  Distance2Node min = minHeapArray[1];
  minHeapArray[1] = minHeapArray[HEAP_SIZE];
  HEAP_SIZE--;
  heapify(minHeapArray, 1);
  return min;
}

/*Get the value of the tree i.e the min key*/
Distance2Node getMin(P2VAR(Distance2Node, HOST) minHeapArray) {
  return minHeapArray[1];
}
