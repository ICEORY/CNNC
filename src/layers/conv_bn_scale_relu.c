#include "conv_bn_scale_relu.h"
#include "utils.h"
#include <math.h>

DataBlob* Ensemble_Convb_BNn_ReLU(DataBlob* bottom, ParamsBlobL *params,
                                  uint in_plane, uint out_plane,
                                  D_Type *conv_weight, D_Type *conv_bias,
                                  D_Type *bn_mean, D_Type *bn_var, D_Type bn_scale_factor,
                                  D_Type *scale_gamma){

    uint n=0, co=0, ci=0, h=0, w=0;
    uchar kh=0, kw=0;
    uint top_index=0, bottom_index, weight_index=0;
    uint top_offset=0, bottom_offset=0, weight_offset=0;
    uint hend=0, wend=0;
    int hstart=0, wstart=0;
    uchar kh_shift=0, kw_shift=0;

    DataBlob *top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    top->n = bottom->n;
    top->c = out_plane;
    top->h = (uint)(floor((float)(bottom->h+2*params->padding_h-params->kernel_h)/params->stride_h))+1;
    top->w = (uint)(floor((float)(bottom->w+2*params->padding_w-params->kernel_w)/params->stride_w))+1;
    top->data = (D_Type*)MemoryPool(sizeof(D_Type)*top->n*top->c*top->h*top->w);

    D_Type temp_mean=0, temp_std=0, temp_gamma=0;
    if (bn_scale_factor!=0){
        bn_scale_factor = 1.0/bn_scale_factor;
    }

    for (n=0;n<bottom->n;n=n+1){
        for (co=0;co<top->c;co=co+1){
            top_offset = n*top->c*top->h*top->w+co*top->h*top->w;
            temp_mean = bn_mean[co]*bn_scale_factor;
            temp_std = 1.0/sqrt(bn_var[co]*bn_scale_factor+0.00001);
            temp_gamma = scale_gamma[co];
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    top_index = top_offset+h*top->w+w;
                    // run convolution
                    for (ci=0;ci<bottom->c;ci=ci+1){
                        bottom_offset = n*bottom->c*bottom->h*bottom->w+ci*bottom->h*bottom->w;
                        weight_offset = co*in_plane*params->kernel_h*params->kernel_w+ci*params->kernel_h*params->kernel_w;
                        hstart = h*params->stride_h-params->padding_h;
                        wstart = w*params->stride_w-params->padding_w;
                        kh_shift = max(0-hstart, 0);
                        kw_shift = max(0-wstart, 0);
                        //printf(">>>kh_shift:%d, kw_shift: %d\n",kh_shift,kw_shift);
                        hend = min(hstart+params->kernel_h, bottom->h);
                        wend = min(wstart+params->kernel_w, bottom->w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        for (kh=hstart;kh<hend;kh=kh+1){
                            for (kw=wstart;kw<wend;kw=kw+1){
                                bottom_index = bottom_offset+kh*bottom->w+kw;
                                weight_index = weight_offset+(kh-hstart+kh_shift)*params->kernel_w+(kw-wstart+kw_shift);
                                //printf(">>>weight_index:%d\n",weight_index);
                                top->data[top_index] = top->data[top_index]+bottom->data[bottom_index]*conv_weight[weight_index];
                            }
                        }
                    }
                    top->data[top_index] = top->data[top_index]+conv_bias[co];
                    top->data[top_index] = (top->data[top_index]-temp_mean)*temp_std*temp_gamma;
                    top->data[top_index] = max(top->data[top_index], 0);
                }
            }
        }
    }
    MemoryFree(bottom->data);
    MemoryFree(bottom);
    //printf(">>>convolution: n:%d, c:%d, h:%d, w:%d\n",top->n, top->c, top->h, top->w);
    return top;
}

DataBlob* Ensemble_Convb_BNn(DataBlob* bottom, ParamsBlobL *params,
                             uint in_plane, uint out_plane,
                             D_Type *conv_weight, D_Type *conv_bias,
                             D_Type *bn_mean, D_Type *bn_var, D_Type bn_scale_factor,
                             D_Type *scale_gamma){

    uint n=0, co=0, ci=0, h=0, w=0;
    uchar kh=0, kw=0;
    uint top_index=0, bottom_index, weight_index=0;
    uint top_offset=0, bottom_offset=0, weight_offset=0;
    uint hend=0, wend=0;
    int hstart=0, wstart=0;
    uchar kh_shift=0, kw_shift=0;

    DataBlob *top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    top->n = bottom->n;
    top->c = out_plane;
    top->h = (uint)(floor((float)(bottom->h+2*params->padding_h-params->kernel_h)/params->stride_h))+1;
    top->w = (uint)(floor((float)(bottom->w+2*params->padding_w-params->kernel_w)/params->stride_w))+1;
    top->data = (D_Type*)MemoryPool(sizeof(D_Type)*top->n*top->c*top->h*top->w);

    D_Type temp_mean=0, temp_std=0, temp_gamma=0;
    if (bn_scale_factor!=0){
        bn_scale_factor = 1.0/bn_scale_factor;
    }

    for (n=0;n<bottom->n;n=n+1){
        for (co=0;co<top->c;co=co+1){
            top_offset = n*top->c*top->h*top->w+co*top->h*top->w;
            temp_mean = bn_mean[co]*bn_scale_factor;
            temp_std = 1.0/sqrt(bn_var[co]*bn_scale_factor+0.00001);
            temp_gamma = scale_gamma[co];
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    top_index = top_offset+h*top->w+w;
                    // run convolution
                    for (ci=0;ci<bottom->c;ci=ci+1){
                        bottom_offset = n*bottom->c*bottom->h*bottom->w+ci*bottom->h*bottom->w;
                        weight_offset = co*in_plane*params->kernel_h*params->kernel_w+ci*params->kernel_h*params->kernel_w;
                        hstart = h*params->stride_h-params->padding_h;
                        wstart = w*params->stride_w-params->padding_w;
                        kh_shift = max(0-hstart, 0);
                        kw_shift = max(0-wstart, 0);
                        //printf(">>>kh_shift:%d, kw_shift: %d\n",kh_shift,kw_shift);
                        hend = min(hstart+params->kernel_h, bottom->h);
                        wend = min(wstart+params->kernel_w, bottom->w);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        for (kh=hstart;kh<hend;kh=kh+1){
                            for (kw=wstart;kw<wend;kw=kw+1){
                                bottom_index = bottom_offset+kh*bottom->w+kw;
                                weight_index = weight_offset+(kh-hstart+kh_shift)*params->kernel_w+(kw-wstart+kw_shift);
                                //printf(">>>weight_index:%d\n",weight_index);
                                top->data[top_index] = top->data[top_index]+bottom->data[bottom_index]*conv_weight[weight_index];
                            }
                        }
                    }
                    top->data[top_index] = top->data[top_index]+conv_bias[co];
                    top->data[top_index] = (top->data[top_index]-temp_mean)*temp_std*temp_gamma;
                }
            }
        }
    }
    MemoryFree(bottom->data);
    MemoryFree(bottom);
    //printf(">>>convolution: n:%d, c:%d, h:%d, w:%d\n",top->n, top->c, top->h, top->w);
    return top;
}
