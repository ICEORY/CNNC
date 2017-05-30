#ifndef _NET_GENERATOR_H_
#define _NET_GENERATOR_H_

#include "utils.h"
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
    uchar top_1_id;
    uchar top_2_id;
    struct LayerNode_ *top_1;
    struct LayerNode_ *top_2;
    D_Type *weight;
}LayerNode;

typedef struct LayerNodeList_{
    LayerNode **node_list;
    uchar node_len;
}LayerNodeList;

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
net describe file format:

convoludion:
format: net_type, layer_id, top_layer, in_plane, out_plane, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w,  bias_term
length: 12

batch normalization:
format: net_type, layer_id, top_layer, in_plane
length: 4

relu:
format: net_type, layer_id, top_layer
length: 3

avg pooling:
format: net_type, layer_id, top_layer, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w
length: 9

max pooling:
format: net_type, layer_id, top_layer, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w
length: 9

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

*/
void NetFileParse(char *file_path, LayerNodeList *node_list);


/**
link all nodes
layer_node_list contains node_list and node_list_len
step 1: get layer_node from node_list and judge whether it has next nodes
step 2: set top_node in layer_node pointing to exact layer_node in node_list;
*/
void LinkNode(LayerNodeList *node_list);


/**
get weight data from file
weight file format:
layer_id data_len data[...]
-------------------------------------------------------

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
void WeightFileParse(char *file_path, LayerNodeList *node_list);


#endif // _NET_GENERATOR_H_
