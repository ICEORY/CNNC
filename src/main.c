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

 //#include "resnet_model.h"
 #include "readfile.h"

 #include "net_generator.h"

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
 }

/**TO DO:
we develop framework with considering of generation and efficiency.
you may need to refactoring these codes for more specific applications, i.e. applying on ARM.
*/
int main(){
    //TestAll();
    uint i=0;
    int *a;
    a = NULL;
    int a_sub[3] = {12,56,45};
    printf("%i\n",a);
    a = a_sub;
    printf("%i\n",a);
    for (i=0;i<3;i=i+1){
        printf("data%d\n",a[i]);
        a[i] = 0;
        printf("..data%d\n",a_sub[i]);
    }

    system("Pause");
    return 0;
}

