#ifndef _BATCH_NORMALIZATION_H_
#define _BATCH_NORMALIZATION_H_

#include <math.h>
#include "utils.h"
/**
\hat{x} = \frac{x-mean}{\square{var-eps}}
y = \hat{x}*\gmma+\beta
*/
void BatchNormalization(const DataBlob *bottom, DataBlob *top,
                        const WeightBlob *gamma, const WeightBlob *beta,
                        const WeightBlob *mean, const WeightBlob *var,
                        D_Type scale_factor, const D_Type eps){
    uint n=0, c=0, h=0, w=0;

    if (scale_factor != 0){
        scale_factor = 1/scale_factor;
    }

    top->n = bottom->n;
    top->c = bottom->c;
    top->h = bottom->h;
    top->w = bottom->w;

    for (n=0;n<bottom->n;n=n+1){
        for (c=0;c<top->c;c=c+1){
            uint top_offset = n*top->c*top->h*top->w+c*top->h*top->w;
            float scale = gamma->data[c]/sqrt(var->data[c]*scale_factor+eps);
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    uint top_index = top_offset+h*top->w+w;
                    top->data[top_index] = (bottom->data[top_index]-mean->data[c]*scale_factor)*scale+beta->data[c];
                }
            }
        }
    }
}

void BatchNormalizationTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)malloc(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 3;
    bottom->w = 3;
    bottom->data = input;

    D_Type gamma_data[1] = {2};
    WeightBlob gamma = {1,1,1,1,gamma_data};
    D_Type beta_data[1] = {-1};
    WeightBlob beta = {1,1,1,1,beta_data};
    D_Type mean_data[1] = {2.55};
    WeightBlob mean = {1,1,1,1,mean_data};
    D_Type var_data[1] = {9.14};
    WeightBlob var = {1,1,1,1,var_data};
    D_Type scale_factor = 9999.8;
    D_Type eps = 0.00001;


    DataBlob *top = (DataBlob *)malloc(sizeof(DataBlob));
    D_Type *top_memory = (D_Type*)malloc(sizeof(D_Type)*9);
    memset(top_memory, 0, sizeof(*top_memory));
    top->data = top_memory;
    uint i= 0;
    for (i=0;i<1;i++){
        BatchNormalization(bottom, top, &gamma, &beta, &mean, &var, scale_factor, eps);
        PrintAll(top);
    }
}
#endif // _BATCH_NORMALIZATION_H_
