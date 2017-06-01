#include "concat.h"
#include "utils.h"
#include <stdio.h>
DataBlob* ConcatTable(DataBlob *bottom, DataBlob *top_2){
    uint top_count = bottom->n * bottom->c * bottom->h * bottom->w;
    uint i =0;

    DataBlob *top_1 = bottom;
    top_2->n = bottom->n;
    top_2->c = bottom->c;
    top_2->h = bottom->h;
    top_2->w = bottom->w;
    top_2->data = (D_Type*)MemoryPool(sizeof(D_Type)*bottom->n*bottom->c*bottom->h*bottom->w);

    for (i=0;i<top_count;i=i+1){
        top_2->data[i] = bottom->data[i];
    }
    //printf(">>>concat table-1: n:%d, c:%d, h:%d, w:%d\n",top_1->n, top_1->c, top_1->h, top_1->w);
    //printf(">>>concat table-2: n:%d, c:%d, h:%d, w:%d\n",top_2->n, top_2->c, top_2->h, top_2->w);
    return top_1;
}

DataBlob *CAddTable(DataBlob *bottom_1, DataBlob *bottom_2){
    uint top_count = bottom_1->n * bottom_1->c * bottom_1->h * bottom_1->w;

    uint i =0;
    DataBlob *top = bottom_1;
    for (i=0;i<top_count;i=i+1){
        top->data[i] = bottom_1->data[i]+bottom_2->data[i];
    }
    MemoryFree(bottom_2->data);
    MemoryFree(bottom_2);
    //printf(">>>add table: n:%d, c:%d, h:%d, w:%d\n",top->n, top->c, top->h, top->w);
    return top;
}

void ConcatTableTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 3;
    bottom->w = 3;
    bottom->data = input;

    DataBlob *top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
    DataBlob* top_1 = ConcatTable(bottom, top_2);
    PrintAll(top_1);
    PrintAll(top_2);
    printf("Test ConcatTable Pass\n");
}

void CAddTableTest(){
    D_Type input_1[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom_1 = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom_1->n = 1;
    bottom_1->c = 1;
    bottom_1->h = 3;
    bottom_1->w = 3;
    bottom_1->data = input_1;

    D_Type input_2[9] = {10, -2, 3, 0, 5, -3, 6, -5, 6};
    DataBlob *bottom_2 = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom_2->n = 1;
    bottom_2->c = 1;
    bottom_2->h = 3;
    bottom_2->w = 3;
    bottom_2->data = input_2;

    DataBlob *top = CAddTable(bottom_1, bottom_2);
    PrintAll(top);
    printf("Test CAddTable Pass\n");
}
