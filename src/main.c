/** \project Convolutional Neural Networks in C: CNN-C
 * \author iceory
 * \email z.zhuangwei@scut.edu.cn
 * \date 2017.5.29
 * \reference: https://github.com/BVLC/caffe
 */

 //==========Include All Layers===========================
 #include "avg_pooling.h"
 #include "batch_normalization.h"
 #include "concat.h"
 #include "convolution.h"
 #include "data_transform.h"
 #include "linear.h"
 #include "max_pooling.h"
 #include "relu.h"
 #include "scale.h"

 #include "resnet_model.h"
 #include "readfile.h"

 #include "net_generator.h"
 #include "net_solver.h"

 void TestAll(){
    AvgPoolingTest();
    BatchNormalizationTest();
    ConcatTableTest();
    CAddTableTest();
    ConvolutionTest();
    CenterCropTest();
    DataNormalizeTest();
    LinearTest();
    MaxPoolingTest();
    ReLUTest();
    ScaleTest();
    ReadDatTest();
    resnet20test();
 }

/**
TO DO:
we develop framework with considering of generation and efficiency.
you may need to refactoring these codes for more specific applications, i.e. applying on ARM.

We also ignore data preprocessing, and you can define them by yourself.

All these codes can be optimized further.

After All, good luck~~
*/


int main(){
    TestAll();
    return 0;
}

