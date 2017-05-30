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

LayerNode* NodeRun(LayerNode *node, DataBlob *bottom, DataBlob *top){
    if (node->net_type == NID_AVG_POOLING)
    {
        ParamsBlobL *params = (ParamsBlobL*)MemoryPool(sizeof(ParamsBlobL));
        params->kernel_h = node->params[0];
        params->kernel_w = node->params[1];
        params->padding_h = node->params[2];
        params->padding_w = node->params[3];
        params->stride_h = node->params[4];
        params->stride_w = node->params[5];

        AvgPooling(bottom, top, params);
        *bottom = *top;
        top = (DataBlob*)MemoryPool(sizeof(DataBlob));
        if (node->node_num == 1){
            NodeRun(node->top_1, bottom, top);
        }
        MemoryFree(params);
    }

    if (node->net_type == NID_BATCH_NORMALIZATION){
        WeightBlob *mean = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
        mean->data = node->weight;

        WeightBlob *var = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
        var->data = node->weight+node->params[0];

        D_Type *scale_factor = node->weight+2*node->params[0];
        BatchNormalization(bottom, top, mean, var, scale_factor[0], BN_EPS);
        *bottom = *top;
        top = (DataBlob*)MemoryPool(sizeof(DataBlob));
        if (node->node_num == 1){
            NodeRun(node->top_1, bottom, top);
        }
        MemoryFree(mean->data);
        MemoryFree(var->data);
        MemoryFree(mean);
        MemoryFree(var);
    }

    //if (node->net_type == NID_CADDTABLE){
        // do nothing
    //}

    if (node->net_type == NID_CONCATTABLE){
        DataBlob *top_1 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        DataBlob *top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        ConcatTable(bottom, top_1, top_2);
        DataBlob *bottom_1 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        *bottom_1 = *top_1;
        DataBlob *bottom_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        bottom_2 = top_2;
        top_1 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        top_2 = (DataBlob*)MemoryPool(sizeof(DataBlob));
        NodeRun(node->top_1, bottom_1, top_1);
        LayerNode *merge_node = NodeRun(node->top_2, bottom_2, top_2);
        *bottom_1 = *top_1;
        *bottom_2 = *top_2;
        CAddTable(bottom_1, bottom_2, top);
        *bottom = *top;
        top = (DataBlob*)MemoryPool(sizeof(DataBlob));
        if (merge_node->node_num == 1){
            NodeRun(merge_node, bottom, top);
        }
    }

    if (node->net_type == NID_CONVOLUTION){

        WeightBlob *weight = (WeightBlob*)MemoryPool(sizeof(WeightBlob));
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

        Convolution(bottom, top, weight, bias, params, bias_term);

        *bottom = *top;
        top = (DataBlob*)MemoryPool(sizeof(DataBlob));
        if (node->node_num == 1){
            NodeRun(node->top_1, bottom, top);
        }
        MemoryFree(params);
        MemoryFree(weight->data);
        MemoryFree(weight);
        if(bias_term){
            MemoryFree(bias->data);
            MemoryFree(bias);
        }
    }

    if (node->net_type == NID_DATA_LAYER){
        // you can add preprocessing program here
        if (node->node_num == 1){
            NodeRun(node->top_1, bottom, top);
        }
    }

    if (node->net_type == NID_LINEAR){
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
        Linear(bottom, top, weight, bias, bias_term);
        *bottom = *top;
        top = (DataBlob*)MemoryPool(sizeof(DataBlob));
        if (node->node_num == 1){
            NodeRun(node->top_1, bottom, top);
        }

        MemoryFree(weight->data);
        MemoryFree(weight);
        if(bias_term){
            MemoryFree(bias->data);
            MemoryFree(bias);
        }
    }

    if (node->net_type == NID_MAX_POOLING){
        ParamsBlobL *params = (ParamsBlobL*)MemoryPool(sizeof(ParamsBlobL));
        params->kernel_h = node->params[0];
        params->kernel_w = node->params[1];
        params->padding_h = node->params[2];
        params->padding_w = node->params[3];
        params->stride_h = node->params[4];
        params->stride_w = node->params[5];

        MaxPooling(bottom, top, params);
        *bottom = *top;
        top = (DataBlob*)MemoryPool(sizeof(DataBlob));
        if (node->node_num == 1){
            NodeRun(node->top_1, bottom, top);
        }
        MemoryFree(params);
    }

    if (node->net_type == NID_RELU){
        ReLU(bottom, top);
        *bottom = *top;
        top = (DataBlob*)MemoryPool(sizeof(DataBlob));
        if (node->node_num == 1){
            NodeRun(node->top_1, bottom, top);
        }
    }

    if (node->net_type == NID_SCALE){
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

        Scale(bottom, top, gamma, beta, bias_term);

        *bottom = *top;
        top = (DataBlob*)MemoryPool(sizeof(DataBlob));
        if (node->node_num == 1){
            NodeRun(node->top_1, bottom, top);
        }
        MemoryFree(gamma->data);
        MemoryFree(gamma);
        MemoryFree(beta->data);
        MemoryFree(beta);
    }

    if (node->net_type == NID_CADDTABLE){
        return node;
    }
    else{
        return NULL;
    }
}


void NetWorkTest(){
    LayerNodeList *node_list = (LayerNodeList*)MemoryPool(sizeof(LayerNodeList));
    NetFileParse("network.dat", node_list);
    LinkNode(node_list);
    WeightFileParse("model.dat", node_list);
    DataBlob *bottom = (DataBlob*)MemoryPool(sizeof(bottom));
    DataBlob *top = (DataBlob*)MemoryPool(sizeof(top));
    NodeRun(node_list->node_list[0], bottom, top);
    PrintAll(top);
}
