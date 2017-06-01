#include "net_solver.h"
#include "net_generator.h"

#include "avg_pooling.h"
#include "batch_normalization.h"
#include "concat.h"
#include "convolution.h"
#include "data_transform.h"
#include "linear.h"
#include "max_pooling.h"
#include "relu.h"
#include "scale.h"

#include "utils.h"
#include <stdio.h>

DataBlob* NodeRun(const LayerNode *node, DataBlob *bottom){
    DataBlob *top;
    if (node->net_type == NID_AVG_POOLING)
    {
        //printf(">>>Enter avg pooling:%d\n", node->layer_id);
        ParamsBlobL *params = (ParamsBlobL*)MemoryPool(sizeof(ParamsBlobL));
        params->kernel_h = node->params[0];
        params->kernel_w = node->params[1];
        params->padding_h = node->params[2];
        params->padding_w = node->params[3];
        params->stride_h = node->params[4];
        params->stride_w = node->params[5];

        top = AvgPooling(bottom, params);

        if (node->node_num == 1){
            bottom = top;
            top = NodeRun(node->top_1, bottom);
        }
        MemoryFree(params);
        return top;
    }

    if (node->net_type == NID_BATCH_NORMALIZATION){
        //printf(">>>Enter batch norm:%d\n", node->layer_id);
        WeightBlob *mean = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
        mean->data = node->weight;

        WeightBlob *var = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
        var->data = node->weight+node->params[0];

        D_Type *scale_factor = node->weight+2*node->params[0];
        top = BatchNormalization(bottom, mean, var, scale_factor[0], BN_EPS);

        if (node->node_num == 1){
            bottom = top;
            top = NodeRun(node->top_1, bottom);
        }
        MemoryFree(mean);
        MemoryFree(var);
        return top;
    }

    if (node->net_type == NID_CONCATTABLE){
        //printf(">>>Enter concat table:%d\n", node->layer_id);
        DataBlob *top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        DataBlob *top_1 = ConcatTable(bottom, top_2);

        DataBlob *bottom_1 = top_1;
        DataBlob *bottom_2 = top_2;

        top_1 = NodeRun(node->top_1, bottom_1);
        top_2 = NodeRun(node->top_2, bottom_2);

        LayerNode *merge_node = (LayerNode*)MemoryPool(sizeof(LayerNode));
        *merge_node = *node;
        while(merge_node->net_type != NID_CADDTABLE){
            //printf("next net_type is:%d\n",merge_node->net_type);
            merge_node = merge_node->top_1;
        }
        bottom_1 = top_1;
        bottom_2 = top_2;
        top = CAddTable(bottom_1, bottom_2);

        if (merge_node!=NULL && merge_node->node_num == 1){
            bottom = top;
            top = NodeRun(merge_node->top_1, bottom);
        }
        MemoryFree(merge_node->params);
        MemoryFree(merge_node->weight);
        MemoryFree(merge_node->top_1);
        MemoryFree(merge_node->top_2);
        MemoryFree(merge_node);
        return top;
    }

    if (node->net_type == NID_CONVOLUTION){
        printf(">>>Enter convolution:%d\n", node->layer_id);
        WeightBlob *weight = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
        //printf("===>params num: %d\n", node->params_num);
        weight->data = node->weight;
        weight->in_plane = node->params[0];
        weight->out_plane = node->params[1];
        weight->kernel_h = node->params[2];
        weight->kernel_w = node->params[3];

        ParamsBlobS *params = (ParamsBlobS*)MemoryPool(sizeof(ParamsBlobS));
        params->padding_h = node->params[4];
        params->padding_w = node->params[5];
        params->stride_h = node->params[6];
        params->stride_w = node->params[7];

        uchar bias_term = node->params[8];

        WeightBlob *bias;
        if (bias_term){
            bias = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
            bias->data = node->weight+weight->in_plane*weight->out_plane*weight->kernel_h*weight->kernel_w;
        }
        else{
            bias = NULL;
        }
        //printf(">>>convolution: n:%d, c:%d, h:%d, w:%d\n",bottom->n, bottom->c, bottom->h, bottom->w);
        top = Convolution(bottom, weight, bias, params, bias_term);

        if (node->node_num == 1){
            bottom = top;
            top = NodeRun(node->top_1, bottom);
        }
        MemoryFree(params);
        MemoryFree(weight);
        if(bias_term){
            MemoryFree(bias);
        }
        return top;
    }

    if (node->net_type == NID_DATA_LAYER){
        printf(">>>Enter data:%d\n", node->layer_id);
        // you can add preprocessing program here
        top = CenterCrop(bottom, 32, 32);
        if (node->node_num == 1){
            bottom = top;
            top = NodeRun(node->top_1, bottom);
        }
        return top;
    }

    if (node->net_type == NID_LINEAR){
        //printf(">>>Enter linear:%d\n", node->layer_id);
        WeightBlob *weight = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
        weight->data = node->weight;
        weight->in_plane = node->params[0];
        weight->out_plane = node->params[1];
        weight->kernel_h = 1;
        weight->kernel_w = 1;
        uchar bias_term = node->params[2];

        WeightBlob *bias;
        if (bias_term){
            bias = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
            bias->data = node->weight+weight->in_plane*weight->out_plane;
        }
        else{
            bias = NULL;
        }
        top = Linear(bottom, weight, bias, bias_term);

        if (node->node_num == 1){
            //printf("linear layer enter final point\n");
            bottom = top;
            top = NodeRun(node->top_1, bottom);
        }
        //PrintAll(top);
        MemoryFree(weight);
        if(bias_term){
            MemoryFree(bias);
        }
        return top;
    }

    if (node->net_type == NID_MAX_POOLING){
        //printf(">>>Enter max pooling:%d\n", node->layer_id);
        ParamsBlobL *params = (ParamsBlobL*)MemoryPool(sizeof(ParamsBlobL));
        params->kernel_h = node->params[0];
        params->kernel_w = node->params[1];
        params->padding_h = node->params[2];
        params->padding_w = node->params[3];
        params->stride_h = node->params[4];
        params->stride_w = node->params[5];

        top = MaxPooling(bottom, params);

        if (node->node_num == 1){
            bottom = top;
            top = NodeRun(node->top_1, bottom);
        }
        MemoryFree(params);
        return top;
    }

    if (node->net_type == NID_RELU){
        //printf(">>>Enter relu:%d\n", node->layer_id);
        top = ReLU(bottom);
        if (node->node_num == 1){
            bottom = top;
            top = NodeRun(node->top_1, bottom);
        }
        return top;
    }

    if (node->net_type == NID_SCALE){
        //printf(">>>Enter scale:%d\n", node->layer_id);
        //printf("===>params num: %d\n", node->params_num);
        WeightBlob *gamma = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
        gamma->data = node->weight;
        uchar bias_term = node->params[1];
        WeightBlob *beta;
        if (bias_term){
            beta = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
            beta->data = node->weight+node->params[0];
        }
        else{
            beta = NULL;
        }

        top = Scale(bottom, gamma, beta, bias_term);

        if (node->node_num == 1){
            bottom = top;
            top = NodeRun(node->top_1, bottom);
        }
        MemoryFree(gamma);
        MemoryFree(beta);
        return top;
    }

    if (node->net_type == NID_CADDTABLE){
        top = bottom;
        return top;
    }
    printf("warning: NULL return:%d\n", node->net_type);
    return NULL;
}


void NetWorkTest(){
    uint i = 0;
    LayerNodeList *node_list = (LayerNodeList*)MemoryPool(sizeof(LayerNodeList));
    NetFileParse("tools/network.dat", node_list);
    LinkNode(node_list);
    WeightFileParse("tools/weight.dat", node_list);
    //printf("data parse done!\n");
    DataBlob *bottom = (DataBlob*)MemoryPool(sizeof(bottom));
    bottom->n = 1;
    bottom->c = 3;
    bottom->h = 32;
    bottom->w = 32;
    bottom->data = (D_Type*)MemoryPool(sizeof(D_Type)*32*32*3);//32*32*3=3072
    for (i=0;i<3072;i=i+1){
        bottom->data[i]= i%10+i*0.02;
    }
    DataBlob *top;
    for (i=0;i<1;i=i+1){
        top = NodeRun(node_list->node_list[0], bottom);
        printf("%d\n",i);
        /*if (top!=NULL){
            MemoryFree(top->data);
            MemoryFree(top);
        }*/
    }
    //PrintAll(top);
    //MemoryFree(top);
}
