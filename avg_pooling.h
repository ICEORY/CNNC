#ifndef _AVG_POOLING_H_
#define _AVG_POOLING_H_
#include <math.h>
#include "utils.h"

void AvgPooling(const DataBlob *bottom, DataBlob *top, const ParamsBlobL *params){
    uint n=0, c=0;
    uchar ph=0, pw=0, h=0, w=0;

    top->n = bottom->n;
    top->c = bottom->c;
    top->h = (uint)(ceil((float)(bottom->h+2*params->padding_h-params->kernel_h)/params->stride_h))+1;
    top->w = (uint)(ceil((float)(bottom->w+2*params->padding_w-params->kernel_w)/params->stride_w))+1;

    for (n=0;n<bottom->n;n=n+1){
        for (c=0;c<bottom->c;c=c+1){
            uint bottom_offset = n*bottom->c*bottom->h*bottom->w + c*bottom->h*bottom->w;
            uint top_offset = n*top->c*top->h*top->w + c*top->h*top->w;
            for (ph=0;ph<top->h;ph=ph+1){
                for (pw=0;pw<top->w;pw=pw+1){

                    int hstart = ph*params->stride_h-params->padding_h;
                    int wstart = pw*params->stride_w-params->padding_w;
                    uint hend = min(hstart+params->kernel_h, bottom->h+params->padding_h);
                    uint wend = min(wstart+params->kernel_w, bottom->w+params->padding_w);
                    uchar pool_size = (hend-hstart)*(wend-wstart);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    hend = min(hend, bottom->h);
                    wend = min(wend, bottom->w);

                    uint pooled_index = top_offset+ph*top->w+pw;
                    for (h=hstart;h<hend;h=h+1){
                        for (w=wstart;w<wend;w=w+1){
                            uint index = bottom_offset+h*bottom->w+w;
                            top->data[pooled_index] = top->data[pooled_index]+bottom->data[index];
                        }
                    }
                    top->data[pooled_index] = top->data[pooled_index]/pool_size;
                }
            }
        }
    }

}


void AvgPoolingTest(){
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
    ParamsBlobL params = {3, 3, 1, 1, 1 ,1};
    for (i=0;i<1;i++){
        AvgPooling(bottom, top, &params);
        //PrintAll(top);
    }
}


#endif // _AVG_POOLING_H_
