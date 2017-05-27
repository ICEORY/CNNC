#include <stdio.h>
#include <math.h>
#include "utils.h"
//#include "relu.h"

int main(){
    uint i=0;
    double j=0.1;
    for (i=0;i<10000000;i++){
        j=exp(1.1)+1.1;
    }
    printf("%f\n",j);
    return 0;
}


