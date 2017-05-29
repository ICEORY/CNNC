/** \project Convolutional Neural Networks in C: CNN-C
 * \author iceory
 * \emal z.zhuangwei@scut.edu.cn
 * \date 2017.5.29
 * \reference: https://github.com/BVLC/caffe
 */

#ifndef _NET_CREATOR_H_
#define _NET_CREATOR_H_

#include "utils.h"
#include <stdio.h>

/**
define net_id
*/
#define NID_CONVOLUTION 0X01
#define NID_BATCH_NORMALIZATION 0X02
#define NID_RELU 0X03
#define NID_AVG_POOLING 0X04
#define NID_MAX_POOLING 0X05
#define NID_SCALE 0X06
#define NID_LINEAR 0X07
#define NID_CADDTABLE 0X08
#define NID_CONCATTABLE 0X09
#define NID_DATA_LAYER 0X0A


/**
params_num: number of parameters described in net_creator.h
*params: save params
node_num: number of next node, caddtable=2
top_1, top_2: pointer for getting next layer
*/

typedef struct LayerNode_{
    uchar net_type;
    uchar layer_id;
    uchar params_num;
    short *params;
    uchar node_num;
    struct LayerNode_ *top_1;
    struct LayerNode_ *top_2;
}LayerNode;


/**
net describe file format:

convoludion:
format: net_type, layer_id, top_layer, in_plane, out_plane, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, bias_term
length: 12

batch normalization:
format: net_type, layer_id, top_layer, in_plane
length: 4

relu:
format: net_type, layer_id, top_layer
length: 3

avg pooling:
format: net_type, layer_id, top_layer, stride_h, stride_w, padding_h, padding_w
length: 7

max pooling:
format: net_type, layer_id, top_layer, stride_h, stride_w, padding_h, padding_w
length: 7

linear:
format: net_type, layer_id, top_layer, in_plane, out_plane, bias_term
length: 6

scale:
format: net_type, layer_id, top_layer, in_plane
length: 4

caddtable:
format: net_type, layer_id, top_layer
length: 3

concattable:
format: net_type, layer_id, top_layer_1, top_layer_2;
length: 4

data layer:
format: net_type, layer_id, top_layer, batch_size, channel, height, width
length: 7

-------------------------------------------------------

weight file format:

convolution:
weight (in_plane*out_plane*kernel_h*kernel_w), bias(0 or out_plane)

batch normalization:
mean(in_plane), var(in_plane), gamma(in_plane), beta(in_plane), scale_factor(1)

relu: non

avg pooling: non

max pooling: non

linear:
weight(in_plane*out_plane), bias(0 or out_plane)

scale:
gamma(in_plane), beta(in_plane)

caddtable: none

concattable: none

*/
short ReadDat(FILE *fp){
    short f_input_data = 0;
    fread(&f_input_data, sizeof(f_input_data), 1, fp);
    return f_input_data;
}

void GetLine(FILE *fp, short *buffer, uchar len){
    uchar i = 0;
    fread(buffer, sizeof(buffer)*len, len, fp);
}

/**
create linked list for describing structure of networks
*/
void CreateNode(FILE *fp, LayerNode *head_node){
    uchar i = 0;
    uchar params_num = 0;
    short f_input_data = 0;
    if(!feof(fp)){
        LayerNode *next_node = (LayerNode*)MemoryPool(sizeof(LayerNode));
        f_input_data = ReadDat(fp);
        next_node->net_type = (uchar)f_input_data;
        next_node->layer_id = (uchar)(ReadDat(fp));
        switch(f_input_data){
            case NID_DATA_LAYER:
                params_num = 4;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node = next_node;
                CreateNode(fp, next_node);
                break;

            case NID_CONVOLUTION:
                params_num = 9;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                break;

            case NID_BATCH_NORMALIZATION:
                params_num = 1;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                break;

            case NID_RELU:
                params_num = 0;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                break;

            case NID_AVG_POOLING:
                params_num = 4;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                break;

            case NID_MAX_POOLING:
                params_num = 4;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                break;

            case NID_SCALE:
                params_num = 1;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                break;

            case NID_CADDTABLE:
                params_num = 0;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                break;

            case NID_CONCATTABLE:
                params_num = 0;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                CreateNode(fp, next_node);
                break;

            case NID_LINEAR:
                params_num = 3;
                next_node->params_num = params_num;
                next_node->params = (short*)MemoryPool(sizeof(short)*params_num);
                GetLine(fp, next_node->params, params_num);
                next_node->node_num = 1;
                head_node->top_1 = next_node;
                CreateNode(fp, next_node);
                break;
            default: break;
        }
    }

}

void NetCreatorGeneral(const *char file_path, LayerNode *head_node){
    FILE *fp;
    fp = fopen(file_path, "r");
    CreateNode(fp, head_node);
}

void CreateLayer(LayerNode *head_node, uchar net_type, uchar *layer_id, uchar params_num, short *net_params){

    LayerNode *next_node = (LayerNode*)MemoryPool(sizeof(LayerNode));
    next_node->net_type = net_type;
    next_node->layer_id = *layer_id;
    *layer_id = *layer_id+1;
    next_node->params_num = params_num;
    next_node->params = net_params;
    next_node->node_num = 0;
    next_node->top_1 = NULL;
    next_node->top_2 = NULL;
    head_node=next_node;
}

#endif // _NET_CREATOR_H_
