#ifndef _LINEAR_H_
#define _LINEAR_H_

#include "utils.h"
/**
input dim: N_i*C_i*1*1
output dim: N_o*C_o*1*1
weight dim: N_w*C_i*C_o
where N_i = N_o = N_w = batch size
*/
void Linear(const DataBlob &bottom, DataBlob &top, const WeightBlob &weight, const WeightBlob &bias, uchar bias_term){
    uint n = 0;
    uint c_i = 0;
    uint c_o = 0;

    top->n = bottom->n;
    top->c = weight->out_plane;
    top->h = 1;
    top->w = 1;

    for (n=0;n<bottom->n;n=n+1){
        uint bottom_offset = n*bottom->c; // this code may put outside the loop for improving efficiency further
        uint top_offset = n*top->c;
        for (c_o=0;c_o<top->c;c_o=c_o+1){
            uint out_index = top_offset+c_o;
            for (c_i=0;c_i<bottom->c;c_i=c_i+1){
                top->data[out_index] = top->data[out_index]+weight->data[c_o*weight->in_plane+c_i]*bottom->data[bottom_offset+c_i];
            }
            if (bias_term){
                top->data[out_index] = top->data[out_index]+bias->data[c_o];
            }
        }
    }
}
#endif // _LINEAR_H_
