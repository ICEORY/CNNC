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

DataBlob* conv3x3_layer(DataBlob *bottom, uint in_plane, uint out_plane, uchar stride, D_Type *weight_data){
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

    DataBlob *top = Convolution(bottom, weight, NULL, params, 0);
    MemoryFree(weight);
    MemoryFree(params);
    return top;
}

DataBlob* conv1x1_layer(DataBlob *bottom, uint in_plane, uint out_plane, uchar stride, D_Type *weight_data){
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

    DataBlob *top = Convolution(bottom, weight, NULL, params, 0);
    MemoryFree(weight);
    MemoryFree(params);
    return top;
}

DataBlob* batch_norm_layer(DataBlob *bottom, D_Type* weight_data){
    D_Type *buffer = weight_data;

    WeightBlob *mean = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    mean->data = buffer;
    WeightBlob *var = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    var->data = buffer+bottom->c;
    D_Type *scale_factor = buffer+2*bottom->c;

    DataBlob *top = BatchNormalization(bottom, mean, var, *scale_factor, 0.00001);
    MemoryFree(mean);
    MemoryFree(var);
    return top;
}

DataBlob* scale_layer(DataBlob *bottom, D_Type* weight_data){
    D_Type *buffer = weight_data;

    WeightBlob *gamma = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    gamma->data = buffer;
    WeightBlob *beta = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    beta->data = buffer+bottom->c;

    DataBlob *top = Scale(bottom, gamma, beta, 1);
    MemoryFree(gamma);
    MemoryFree(beta);
    return top;
}

DataBlob* linear_layer(DataBlob *bottom, D_Type* weight_data){
    D_Type *buffer = weight_data;

    WeightBlob *weight = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    weight->data = buffer;
    WeightBlob *bias = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
    bias->data = buffer+weight->in_plane*weight->out_plane;

    DataBlob *top = Linear(bottom, weight, bias, 1);
    MemoryFree(weight);
    MemoryFree(bias);
    return top;
}

DataBlob* avg_pooling_layer(DataBlob *bottom){
    ParamsBlobL *params = (ParamsBlobL*)MemoryPool(sizeof(ParamsBlobL));
    params->kernel_h = 8;
    params->kernel_w = 8;
    params->padding_h = 0;
    params->padding_w = 0;
    params->stride_h = 1;
    params->stride_w = 1;
    DataBlob *top = AvgPooling(bottom, params);
    return top;
}


DataBlob* ResidualBranch(DataBlob *bottom,
                    uint in_plane, uint out_plane, uchar stride,
                    D_Type **weight, uchar ptr_offset){
    DataBlob *top = conv3x3_layer(bottom, in_plane, out_plane, stride, weight[ptr_offset]);
    bottom = top;

    top = batch_norm_layer(bottom, weight[ptr_offset+1]);
    bottom = top;

    top = ReLU(bottom);
    bottom = top;

    top = conv3x3_layer(bottom, out_plane, out_plane, 1, weight[ptr_offset+2]);
    bottom = top;

    top = batch_norm_layer(bottom, weight[ptr_offset+3]);

    return top;
}

DataBlob* ResidualBlock(DataBlob *bottom, uint in_plane, uint out_plane, D_Type **weight, uchar down_sample_term, uchar ptr_offset){

    DataBlob *top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
    DataBlob *top_1 = ConcatTable(bottom,  top_2);
    DataBlob *bottom_1 = top_1;
    DataBlob *bottom_2 = top_2;

    top_1 = ResidualBranch(bottom_1, in_plane, out_plane, 1, weight, ptr_offset);
    bottom_1 = top_1;

    if (down_sample_term){
        top_2 = conv1x1_layer(bottom_2, in_plane, out_plane, 2, weight[ptr_offset+4]);
        bottom_2 = top_2;

        top_2 = batch_norm_layer(bottom_2, weight[ptr_offset+5]);
        bottom_2 = top_2;
    }

    DataBlob *top = CAddTable(bottom_1, bottom_2);
    bottom = top;
    top = ReLU(bottom);
    return top;
}


DataBlob* ResNet_20(DataBlob *bottom, D_Type **weight){
    // data transform
    DataBlob *top = CenterCrop(bottom, 32, 32);
    bottom = top;

    D_Type mean[3] = {0.123, 0.123, 0.123};
    D_Type std_dev[3] = {0.123, 0.123, 0.123};
    top = DataNormalize(bottom, mean, std_dev);
    bottom = top;

    // head convolutional layer
    top = conv3x3_layer(bottom, 3, 16, 1, data_weight_list[0]);
    bottom = top;

    top = batch_norm_layer(bottom, data_weight_list[1]);
    bottom = top;

    top = ReLU(bottom);
    bottom = top;

//================================================================================
    // block 1
    top = ResidualBlock(bottom, 16, 16, weight, 0, 2);
    bottom = top;

    top = ResidualBlock(bottom, 16, 16, weight, 0, 6);
    bottom = top;

    top = ResidualBlock(bottom, 16, 16, weight, 0, 10);
    bottom = top;
//--------------------------
    top = ResidualBlock(bottom, 32, 32, weight, 1, 14);
    bottom = top;

    top = ResidualBlock(bottom, 32, 32, weight, 0, 20);
    bottom = top;

    top = ResidualBlock(bottom, 32, 32, weight, 0, 24);
    bottom = top;

//--------------------------
    top = ResidualBlock(bottom, 64, 64, weight, 1, 28);
    bottom = top;

    top = ResidualBlock(bottom, 64, 64, weight, 0, 34);
    bottom = top;

    top = ResidualBlock(bottom, 64, 64, weight, 0, 38);
    bottom = top;

    top = avg_pooling_layer(bottom);
    bottom = top;

    top = linear_layer(bottom, weight[42]);
    return top;
}

void resnet20test(){
    uint iter = 0, i=0;
    for(iter=0;iter<1;iter=iter+1){
        DataBlob *bottom = (DataBlob*)MemoryPool(sizeof(bottom));
        bottom->n = 1;
        bottom->c = 3;
        bottom->h = 32;
        bottom->w = 32;
        bottom->data = (D_Type*)MemoryPool(sizeof(D_Type)*32*32*3);//32*32*3=3072
        for (i=0;i<3072;i=i+1){
            bottom->data[i]= i%10+i*0.02;
        }
        DataBlob *top = ResNet_20(bottom, data_weight_list);
        PrintAll(top);
        MemoryFree(top);
    }
}
