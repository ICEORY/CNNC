#ifndef _NET_SOLVER_H_
#define _NET_SOLVER_H_
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

/**
node run: run every node on node_list
input: LayerNode *node, DataBlob *bottom, DataBlob *top
return: top feature maps
*/
DataBlob* NodeRun(const LayerNode *node, DataBlob *bottom);

/**
test net run
*/
void NetWorkTest();


#endif // _NET_SOLVER_H_
