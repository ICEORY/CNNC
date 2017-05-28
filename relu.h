#ifndef _RELU_H_
#define _RELU_H_
#include <math.h>
#include <malloc.h>
#include "utils.h"
#include "string.h"

/**
ReLU: y = max(x,0);
*/

void ReLU(const DataBlob *bottom, DataBlob *top){

    uint top_count = bottom->n * bottom->c * bottom->h * bottom->w;
    uint i =0;

    top->n = bottom->n;
    top->c = bottom->c;
    top->h = bottom->h;
    top->w = bottom->w;

    for (i=0;i<top_count;i=i+1){
        top->data[i] = max(bottom->data[i], 0);
    }
}


/**
test relu layer
state: pass
*/
void ReLUTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)malloc(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 3;
    bottom->w = 3;
    bottom->data = input;

    DataBlob *top = (DataBlob *)malloc(sizeof(DataBlob));
    D_Type *top_memory = (D_Type*)malloc(sizeof(D_Type)*9);
    memset(top_memory, 0, sizeof(*top_memory));
    top->data = top_memory;
    uint i= 0;
    for (i=0;i<1000;i++){
        ReLU(bottom, top);
    }
}

#endif // _RELU_H_
