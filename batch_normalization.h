#ifndef _BATCH_NORMALIZATION_H_
#define _BATCH_NORMALIZATION_H_

#include <math.h>
#include "utils.h"
/**
\hat{x} = \frac{x-mean}{\square{var-eps}}
y = \hat{x}*\gmma+\beta
*/
void BatchNormalization(const DataBlob &bottom, DataBlob &top,
                        const WeightBlob &gamma, const WeightBlob &beta,
                        const WeightBlob &mean, const WeightBlob &var,
                        float scale_factor, const float eps){
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
                    uint top_index = top_offset+h*top->h+top->w;
                    top->data[top_index] = (bottom[top_index]-mean->data[c])*scale+beta->data[c];
                }
            }
        }
    }
}
#endif // _BATCH_NORMALIZATION_H_
