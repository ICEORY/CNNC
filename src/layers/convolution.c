#include "convolution.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>

void Convolution(DataBlob *bottom, DataBlob *top,
                 const WeightBlob *weight, const WeightBlob *bias,
                 const ParamsBlobS *params, const uchar bias_term){

    uint n=0, co=0, ci=0, h=0, w=0;
    uchar kh=0, kw=0;

    top->n = bottom->n;
    top->c = weight->out_plane;
    top->h = (uint)(ceil((float)(bottom->h+2*params->padding_h-weight->kernel_h)/params->stride_h))+1;
    top->w = (uint)(ceil((float)(bottom->w+2*params->padding_w-weight->kernel_w)/params->stride_w))+1;
    top->data = (D_Type*)MemoryPool(sizeof(D_Type)*top->n*top->c*top->h*top->w);

    for (n=0;n<bottom->n;n=n+1){
        for (co=0;co<top->c;co=co+1){
            uint top_offset = n*top->c*top->h*top->w+co*top->h*top->w;
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    uint top_index = top_offset+h*top->w+w;
                    for (ci=0;ci<bottom->c;ci=ci+1){
                        uint bottom_offset = n*bottom->c*bottom->h*bottom->w+ci*bottom->h*bottom->w;
                        uint weight_offset = co*weight->in_plane*weight->kernel_h*weight->kernel_w+ci*weight->kernel_h*weight->kernel_w;
                        int hstart = h*params->stride_h-params->padding_h;
                        int wstart = w*params->stride_w-params->padding_w;
                        uchar kh_shift = max(0-hstart, 0);
                        uchar kw_shift = max(0-wstart, 0);
                        //printf(">>>kh_shift:%d, kw_shift: %d\n",kh_shift,kw_shift);
                        uint hend = min(hstart+weight->kernel_h, bottom->h);
                        uint wend = min(wstart+weight->kernel_w, bottom->w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        for (kh=hstart;kh<hend;kh=kh+1){
                            for (kw=wstart;kw<wend;kw=kw+1){
                                uint bottom_index = bottom_offset+kh*bottom->w+kw;
                                uint weight_index = weight_offset+(kh-hstart+kh_shift)*weight->kernel_w+(kw-wstart+kw_shift);
                                //printf(">>>weight_index:%d\n",weight_index);
                                top->data[top_index] = top->data[top_index]+bottom->data[bottom_index]*weight->data[weight_index];
                            }
                        }
                    }
                    if (bias_term){
                        top->data[top_index] = top->data[top_index]+bias->data[co];
                    }
                }
            }
        }
    }
    MemoryFree(bottom);
}


void ConvolutionTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 3;
    bottom->w = 3;
    bottom->data = input;

    D_Type weight_data[9] = {2, 4, -1, 3, 9, -6, 7, 2, 4};
    WeightBlob weight = {1,1,3,3,weight_data};
    D_Type bias_data[1] = {-1};
    WeightBlob bias = {1,1,1,1,bias_data};
    ParamsBlobS params = {1,1,1,1};

    DataBlob *top = (DataBlob *)MemoryPool(sizeof(DataBlob));
    Convolution(bottom, top, &weight, &bias, &params, 1);
    PrintAll(top);
    printf("Test Convolution Pass\n");
}
