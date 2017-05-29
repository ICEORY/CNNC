/** \project Convolutional Neural Networks in C: CNN-C
 * \author iceory
 * \emal z.zhuangwei@scut.edu.cn
 * \date 2017.5.29
 * \reference: https://github.com/BVLC/caffe
 */

 #include "avg_pooling.h"
 #include "batch_normalization.h"
 #include "concat.h"
 #include "convolution.h"
 #include "data_transform.h"
 #include "linear.h"
 #include "max_pooling.h"
 #include "relu.h"
 #include "scale.h"

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
 }

/**TO DO:
we develop framework with considering of generation and efficiency.
you may need to refactoring these codes for more specific applications, i.e. applying on ARM.
*/
int main(){
    TestAll();
    system("Pause");
    return 0;
}


