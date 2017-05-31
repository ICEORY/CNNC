#include "scale.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>

DataBlob* Scale(DataBlob *bottom,
           const WeightBlob *gamma, const WeightBlob *beta, const uchar bias_term){

    uint n=0, c=0, h=0, w=0;

    DataBlob *top = bottom;
    for (n=0;n<bottom->n;n=n+1){
        for (c=0;c<top->c;c=c+1){
            uint top_offset = n*top->c*top->h*top->w+c*top->h*top->w;
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    uint top_index = top_offset+h*top->w+w;
                    if (bias_term){
                        top->data[top_index] = bottom->data[top_index]*gamma->data[c]+beta->data[c];
                    }
                    else{
                        top->data[top_index] = bottom->data[top_index]*gamma->data[c];
                    }

                }
            }
        }
    }
    //printf(">>>scale: n:%d, c:%d, h:%d, w:%d\n",top->n, top->c, top->h, top->w);
    return top;

}


void ScaleTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 3;
    bottom->w = 3;
    bottom->data = input;

    D_Type gamma_data[1] = {2};
    D_Type beta_data[1] = {-1};
    WeightBlob gamma = {1,1,1,1,gamma_data};
    WeightBlob beta = {1,1,1,1,beta_data};

    DataBlob *top = Scale(bottom, &gamma, &beta, 1);
    PrintAll(top);
    printf("Test DataNormalize Pass\n");
}
