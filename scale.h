#ifndef _SCALE_H_
#define _SCALE_H_
#include <math.h>
#include "utils.h"

void Scale(DataBlob *bottom, DataBlob *top,
           const WeightBlob *gamma, const WeightBlob *beta){

    uint n=0, c=0, h=0, w=0;

    top = bottom;
    for (n=0;n<bottom->n;n=n+1){
        for (c=0;c<top->c;c=c+1){
            uint top_offset = n*top->c*top->h*top->w+c*top->h*top->w;
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    uint top_index = top_offset+h*top->w+w;
                    top->data[top_index] = bottom->data[top_index]*gamma->data[c]+beta->data[c];
                }
            }
        }
    }
}


void ScaleTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)malloc(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 3;
    bottom->w = 3;
    bottom->data = input;

    D_Type gamma_data[1] = {2};
    D_Type beta_data[1] = {-1};
    WeightBlob gamma = {1,1,1,1,gamma_data};
    WeightBlob beta = {1,1,1,1,beta_data};

    DataBlob *top = (DataBlob *)malloc(sizeof(DataBlob));
    D_Type *top_memory = (D_Type*)malloc(sizeof(D_Type)*9);
    memset(top_memory, 0, sizeof(*top_memory));
    top->data = top_memory;
    uint i= 0;
    for (i=0;i<1;i++){
        Scale(bottom, top, &gamma, &beta);
        PrintAll(top);
    }
}


#endif // _SCALE_H_
