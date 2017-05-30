#include "resnet_model.h"

#include "avg_pooling.h"
#include "batch_normalization.h"
#include "concat.h"
#include "convolution.h"
#include "data_transform.h"
#include "linear.h"
#include "max_pooling.h"
#include "relu.h"
#include "scale.h"
#include "weight_file.h"

#include "utils.h"
/*
void ReadWeight(char *file_path, D_Type *data, uint weight_count){
    FILE *fp;
    fp = fopen(file_path, "r");
    fread(data, sizeof(D_Type)*weight_count, weight_count, fp);
}*/

void conv3x3_layer(DataBlob *bottom, DataBlob *top, uint in_plane, uint out_plane, uchar stride, D_Type *weight_data){
    WeightBlob *weight = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    weight->in_plane = in_plane;
    weight->out_plane = out_plane;
    weight->kernel_h = 3;
    weight->kernel_w = 3;
    weight->data = weight_data;
    //ReadWeight(weight_file, weight->data, in_plane*out_plane*9);

    ParamsBlobS *params = (ParamsBlobS*)MemoryPool(sizeof(ParamsBlobS));
    params->padding_h = 1;
    params->padding_w = 1;
    params->stride_h = stride;
    params->stride_w = stride;

    Convolution(bottom, top, weight, NULL, params, 0);
}

void conv1x1_layer(DataBlob *bottom, DataBlob *top, uint in_plane, uint out_plane, uchar stride, D_Type *weight_data){
    WeightBlob *weight = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    weight->in_plane = in_plane;
    weight->out_plane = out_plane;
    weight->kernel_h = 1;
    weight->kernel_w = 1;
    weight->data = weight_data;
    //ReadWeight(weight_file, weight->data, in_plane*out_plane*9);

    ParamsBlobS *params = (ParamsBlobS*)MemoryPool(sizeof(ParamsBlobS));
    params->padding_h = 0;
    params->padding_w = 0;
    params->stride_h = stride;
    params->stride_w = stride;

    Convolution(bottom, top, weight, NULL, params, 0);
}

void batch_norm_layer(DataBlob *bottom, DataBlob *top, const D_Type* weight_data){
    D_Type eps = 0.00001;
    D_Type *buffer;
    buffer = weight_data;

    WeightBlob *mean = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    mean->data = buffer;
    WeightBlob *var = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    var->data = buffer+bottom->c;
    WeightBlob *gamma = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    gamma->data = buffer+2*bottom->c;
    WeightBlob *beta = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    beta->data = buffer+3*bottom->c;


    D_Type *scale_factor = buffer+4*bottom->c;
    BatchNormalization(bottom, top, mean, var, gamma, beta, *scale_factor, eps);
}

void linear_layer (DataBlob *bottom, DataBlob *top, D_Type* weight_data){
    D_Type *buffer;
    buffer = weight_data;

    WeightBlob *weight = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    weight->data = buffer;
    WeightBlob *bias = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    bias->data = buffer+weight->in_plane*weight->out_plane;
    Linear(bottom, top, weight, bias, 1);
}


void ResidualBranch(DataBlob *bottom, DataBlob *top,
                    uint in_plane, uint out_plane, uchar stride,
                    D_Type **weight){
    DataBlob *temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    conv3x3_layer(bottom, temp_top, in_plane, out_plane, stride, weight[0]);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    batch_norm_layer(bottom, temp_top, weight[1]);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ReLU(bottom, temp_top);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    conv3x3_layer(bottom, temp_top, out_plane, out_plane, 1, weight[2]);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    batch_norm_layer(bottom, temp_top, weight[3]);
    bottom = temp_top;
    top = temp_top;
}

void ResidualBlock(DataBlob *bottom, DataBlob *top, uint in_plane, uint out_plane, D_Type **weight, uchar down_sample_term){

    DataBlob *temp_top_1 = (DataBlob*)MemoryPool(sizeof(DataBlob));
    DataBlob *temp_top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ConcatTable(bottom, temp_top_1, temp_top_2);
    DataBlob *bottom_1;
    DataBlob *bottom_2;

    bottom_1 = temp_top_1;
    bottom_2 = temp_top_2;

    temp_top_1 = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBranch(bottom_1, temp_top_1, in_plane, out_plane, 1, weight);
    bottom_1 = temp_top_1;

    if (down_sample_term){
        temp_top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        conv1x1_layer(bottom_2, temp_top_2, in_plane, out_plane, 2, *(weight+4));
        bottom_2 = temp_top_2;

        temp_top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        batch_norm_layer(bottom_2, temp_top_2, *(weight+5));
        bottom_2 = temp_top_2;
    }

    DataBlob *temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    CAddTable(bottom_1, bottom_2, temp_top);
    temp_top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ReLU(bottom_2, temp_top_2);
    bottom_2 = temp_top_2;
    top = temp_top;
}


void ResNet_20(DataBlob *bottom, DataBlob *top, D_Type **weight){
    // data transform
    DataBlob *temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    CenterCrop(bottom, temp_top, 32, 32);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    D_Type mean[3] = {0.123, 0.123, 0.123};
    D_Type std_dev[3] = {0.123, 0.123, 0.123};
    DataNormalize(bottom, temp_top, mean, std_dev);
    bottom = temp_top;

    // head convolutional layer
    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    conv3x3_layer(bottom, temp_top, 3, 16, 1, data_weight_list[0]);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    batch_norm_layer(bottom, temp_top, data_weight_list[1]);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ReLU(bottom, temp_top);
    bottom = temp_top;

//================================================================================
    // block 1
    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 16, 16, weight+2, 0);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 16, 16, weight+6, 0);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 16, 16, weight+10, 0);
    bottom = temp_top;
//--------------------------
    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 32, 32, weight+14, 1);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 32, 32, weight+20, 0);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 32, 32, weight+24, 0);
    bottom = temp_top;
//--------------------------
    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 64, 64, weight+28, 1);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 64, 64, weight+34, 0);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    ResidualBlock(bottom, temp_top, 64, 64, weight+38, 0);
    bottom = temp_top;

    temp_top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    linear_layer(bottom, temp_top, *(weight+42));
    top = temp_top;
}
