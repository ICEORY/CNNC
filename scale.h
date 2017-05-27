#ifndef _SCALE_H_
#define _SCALE_H_
#include <math.h>
#include "utils.h"

void Scale(const DataBlob &bottom, DataBlob &top, const WeightBlob &gamma, const WeightBlob &beta){

    uint n=0, c=0, h=0, w=0;

    top->n = bottom->n;
    top->c = bottom->c;
    top->h = bottom->h;
    top->w = bottom->w;

    for (n=0;n<bottom->n;n=n+1){
        for (c=0;c<top->c;c=c+1){
            uint top_offset = n*top->c*top->h*top->w+c*top->h*top->w;
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    uint top_index = top_offset+h*top->h+top->w;
                    top->data[top_index] = bottom[top_index]*gamma->data[c]+beta->data[c];
                }
            }
        }
    }
}


#endif // _SCALE_H_
