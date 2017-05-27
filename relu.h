#ifndef _RELU_H_
#define _RELU_H_
#include "utils.h"

void ReLU(const DataBlob &bottom, DataBlob &top){
    uint input_size = bottom->n * bottom->c * bottom->h * bottom->w

    top->n = bottom->n;
    top->c = bottom->c;
    top->h = bottom->h;
    top->w = bottom->w;
    for (i=0;i<input_size;i=i+1){
        if (input[i]>0){
            top->data[i] = bottom->data[i]
        }
        else{
            top->data[i] = 0;
        }
    }
}

#endif // _RELU_H_
