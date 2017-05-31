#include "linear.h"
#include "utils.h"
#include <stdio.h>
/**
input dim: N_i*C_i*1*1
output dim: N_o*C_o*1*1
weight dim: N_w*C_i*C_o
where N_i = N_o = N_w = batch size
*/
DataBlob* Linear(DataBlob *bottom,
            const WeightBlob *weight, const WeightBlob *bias,
            const uchar bias_term){
    uint n = 0;
    uint c_i = 0;
    uint c_o = 0;

    DataBlob *top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    top->n = bottom->n;
    top->c = weight->out_plane;
    top->h = 1;
    top->w = 1;
    top->data = (D_Type*)MemoryPool(sizeof(D_Type)*top->n*top->c);

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
    MemoryFree(bottom->data);
    MemoryFree(bottom);
    //printf(">>>n:%d, c:%d, h:%d, w:%d\n",top->n, top->c, top->h, top->w);
    return top;
    //PrintAll(top);
}


void LinearTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 9;
    bottom->h = 1;
    bottom->w = 1;
    bottom->data = input;

    D_Type weight_data[18] = {2, 4, -1, 3, 9, -6, 7, 2, 4, -1, 3, 9, -6, 7, 0, 8, 0, 8};
    WeightBlob weight = {9,2,1,1,weight_data};
    D_Type bias_data[2] = {-1, 8};
    WeightBlob bias = {1,2,1,1,bias_data};

    DataBlob *top = Linear(bottom, &weight, &bias, 1);
    PrintAll(top);
    printf("Test Linear Pass\n");
}
