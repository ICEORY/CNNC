#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_
#include <math.h>
#include "utils.h"

void Convolution(const DataBlob &bottom, DataBlob &top, const WeightBlob &weight, const WeightBlob &bias,const ParamsBlobS &params, uchar bias_flag){

    uint n=0, co=0, ci=0, h=0, w=0;
    uchar kh=0, kw=0;

    top->n = bottom->n;
    top->c = weight->out_plane;
    top->h = static_cast<uint>(ceil(static_cast<float>(bottom->h+2*params->padding_h-weight->kernel_h)/params->stride_h))+1;
    top->w = static_cast<uint>(ceil(static_cast<float>(bottom->w+2*params->padding_w-weight->kernel_w)/params->stride_w))+1;

    for (n=0;n<bottom->n;n=n+1){
        for (co=0;co<top->c;co=co+1){
            uint top_offset = n*top->c*top->h*top->w+co*top->h*top->w;
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    uint top_index = top_offset+h*top->w+w;
                    for (ci=0;ci<bottom->c;ci=ci+1){
                        uint bottom_offset = n*bottom->c*bottom->h*bottom->w + ci*bottom->h*bottom->w;
                        uint weight_offset = co*weight->in_plane->weight->kernel_h*weight->kernel_w + ci*weight->kernel_h*weight->kernel_w;
                        int hstart = h*params->stride_h-params->padding_h;
                        int wstart = w*params->stride_w-params->padding_w;
                        uchar kh_shift = max(0-hstart, 0);
                        uchar kw_shift = max(0-wstart, 0);
                        uint hend = min(hstart+weight->kernel_h, bottom->h);
                        uint wend = min(wstart+weight->kernel_w, bottom->w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        for (kh=hstart;kh<hend;kh=kh+1){
                            for (kw=wstart;kw<wend;kw=kw+1){
                                uint bottom_index = bottom_offset+kh*bottom->h+kw;
                                uint weight_index = weight_offset+(hend-kh-1+kh_shift)*weight->kernel_h+(wend-kw-1+kw_shift)
                                top->data[top_index] = top->data[top_index]+bottom->data[bottom_index]*weight->data[weight_index];
                            }
                        }
                    }
                    if (bias_flag){
                        top->data[top_index] = top->data[top_index]+bias->data[co];
                    }
                }
            }
        }
    }
}

#endif // _CONVOLUTION_H_
