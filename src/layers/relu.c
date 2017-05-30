#include "relu.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>

/**
ReLU: y = max(x,0);
*/

void ReLU(DataBlob *bottom, DataBlob *top){

    uint top_count = bottom->n * bottom->c * bottom->h * bottom->w;
    uint i =0;
    MemoryFree(top);
    *top = *bottom;
    for (i=0;i<top_count;i=i+1){
        if (top->data[i]<0){
            top->data[i] = 0;
        }
    }
}


/**
test relu layer
*/
void ReLUTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 3;
    bottom->w = 3;
    bottom->data = input;

    DataBlob *top = (DataBlob *)MemoryPool(sizeof(DataBlob));
    //D_Type *top_memory = (D_Type*)MemoryPool(sizeof(D_Type)*9);
    //memset(top_memory, 0, sizeof(*top_memory));
    //top->data = top_memory;
    ReLU(bottom, top);
    PrintAll(top);
    printf("ReLUTest Pass\n");
}
