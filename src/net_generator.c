#include "net_generator.h"
#include "readfile.h"
#include "utils.h"
#include <stdio.h>

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
format: net_type, layer_id, top_layer, in_plane, bias_term
length: 5

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
mean(in_plane), var(in_plane), scale_factor(1)

relu: non

avg pooling: non

max pooling: non

linear:
weight(in_plane*out_plane), bias(0 or out_plane)

scale:
gamma(in_plane), beta(0 or in_plane)

caddtable: none

concattable: none

*/
void NetFileParse(char *file_path, LayerNodeList *node_list){
    FILE *fp;
    uchar i = 0;
    fp = fopen(file_path, "r");
    node_list->node_len = ReadDatChar(fp);
    node_list->node_list = (LayerNode**)MemoryPool(sizeof(LayerNode*)*(node_list->node_len));
    uchar net_type = 0;
    //printf("node length:%d\n", node_list->node_len);
    while(!feof(fp)){
        net_type = ReadDatChar(fp);
        LayerNode *net_node = (LayerNode*)MemoryPool(sizeof(LayerNode));
        uint *params;
        net_node->net_type = net_type;

        net_node->top_1 = NULL;
        net_node->top_2 = NULL;
        net_node->top_2_id = 0;
        net_node->node_num = 1;
        net_node->weight = NULL;

        net_node->layer_id = ReadDatChar(fp);
        net_node->top_1_id = ReadDatChar(fp);
        //printf("net type:%d, layer_id:%d, top_1:%d", net_node->net_type, net_node->layer_id, net_node->top_1_id);

        if (net_node->top_1_id >= node_list->node_len){
            net_node->node_num = 0;
        }

        switch(net_type){
            case NID_AVG_POOLING:
                net_node->params_num = 6;
                break;

            case NID_BATCH_NORMALIZATION:
                net_node->params_num = 1;
                break;

            case NID_CADDTABLE:
                net_node->params_num = 0;
                break;

            case NID_CONCATTABLE:
                net_node->top_2_id = ReadDatChar(fp);
                //printf("top_2: %d ", net_node->top_2_id);
                net_node->node_num = 2;
                net_node->params_num = 0;
                break;

            case NID_CONVOLUTION:
                net_node->params_num = 9;
                break;

            case NID_DATA_LAYER:
                net_node->params_num = 4;
                break;

            case NID_LINEAR:
                net_node->params_num = 3;
                break;

            case NID_MAX_POOLING:
                net_node->params_num = 6;
                break;

            case NID_RELU:
                net_node->params_num = 0;
                break;

            case NID_SCALE:
                net_node->weight = NULL;
                net_node->params_num = 2;
                break;

            default: break;
        }

        params = (uint*)MemoryPool(sizeof(uint)*net_node->params_num);
        for (i=0;i<net_node->params_num;i=i+1){
            params[i] = ReadDatUInt(fp);
            //printf(" %d ", params[i]);
        }
        net_node->params = params;
        node_list->node_list[net_node->layer_id] = net_node;
        //printf("=========================\n");
    }
    fclose(fp);
    //printf(">>>parse network done! #node=%d\n", node_list->node_len);
    //system("pause");
}
/**
link all nodes
layer_node_list contains node_list and node_list_len
step 1: get layer_node from node_list and judge whether it has next nodes
step 2: set top_node in layer_node pointing to exact layer_node in node_list;
*/
void LinkNode(LayerNodeList *node_list){
    uchar i=0;
    uchar node_num = 0;
    for (i=0;i<node_list->node_len;i=i+1){
        node_num = node_list->node_list[i]->node_num;
        //printf("#top node%d\n", node_num);
        switch (node_num){
            case 1:
                node_list->node_list[i]->top_1 = node_list->node_list[node_list->node_list[i]->top_1_id];
                //printf("next node:%d\n", node_list->node_list[i]->top_1_id);
                break;
            case 2:
                node_list->node_list[i]->top_1 = node_list->node_list[node_list->node_list[i]->top_1_id];
                node_list->node_list[i]->top_2 = node_list->node_list[node_list->node_list[i]->top_2_id];
                //printf("next node_1:%d, node_2:%d\n", node_list->node_list[i]->top_1_id, node_list->node_list[i]->top_2_id);
                break;
            default: break;
        }
    }
    //system("pause");
}

/**
get weight data from file
*/
void WeightFileParse(char *file_path, LayerNodeList *node_list){
    FILE *fp;
    fp = fopen(file_path, "r");
    uint i = 0;
    uchar layer_id = 0;
    uint data_len = 0;
    while(!feof(fp)){
        layer_id = ReadDatChar(fp);
        data_len = ReadDatUInt(fp);
        //printf("layer_id:%d\t data_len:%d\t", layer_id, data_len);
        node_list->node_list[layer_id]->weight = (D_Type*)MemoryPool(sizeof(D_Type)*data_len);
        for (i=0;i<data_len;i=i+1){
            node_list->node_list[layer_id]->weight[i] = ReadDatDType(fp);
            //printf("%f\t", node_list->node_list[layer_id]->weight[i]);
        }
        //printf("\n");
    }
    fclose(fp);
    //system("pause");
}


