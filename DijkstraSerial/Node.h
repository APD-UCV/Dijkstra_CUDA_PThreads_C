
#ifndef NODE_H
#define NODE_H


typedef struct node{
    int node_id;
    int no_neighbors;
    int** adj_list;
} Node;


#endif // !NODE_H
