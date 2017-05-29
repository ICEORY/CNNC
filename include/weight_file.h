#ifndef _WEIGHT_FILE_H_
#define _WEIGHT_FILE_H_

#include "utils.h"

/**
this file only for saving data of weights
name format: DATA_netType_blockID_layerID, i.e. DATA_CONV_1_1
*/

/**example for resnet-20*/
//head block
const D_Type DATA_CONV_0_0[1] = {0.1
};
const D_Type DATA_BN_0_0[1] = {0.1
};

//block 1
const D_Type DATA_CONV_1_0[1] = {0.1
};
const D_Type DATA_BN_1_0[1] = {0.1
};
const D_Type DATA_CONV_1_1[1] = {0.1
};
const D_Type DATA_BN_1_1[1] = {0.1
};

//block 2
const D_Type DATA_CONV_2_0[1] = {0.1
};
const D_Type DATA_BN_2_0[1] = {0.1
};
const D_Type DATA_CONV_2_1[1] = {0.1
};
const D_Type DATA_BN_2_1[1] = {0.1
};

//block 3
const D_Type DATA_CONV_3_0[1] = {0.1
};
const D_Type DATA_BN_3_0[1] = {0.1
};
const D_Type DATA_CONV_3_1[1] = {0.1
};
const D_Type DATA_BN_3_1[1] = {0.1
};


//block 4
const D_Type DATA_CONV_4_0[1] = {0.1
};
const D_Type DATA_BN_4_0[1] = {0.1
};
const D_Type DATA_CONV_4_1[1] = {0.1
};
const D_Type DATA_BN_4_1[1] = {0.1
};
const D_Type DATA_CONV_4_2[1] = {0.1
};
const D_Type DATA_BN_4_2[1] = {0.1
};

//block 5
const D_Type DATA_CONV_5_0[1] = {0.1
};
const D_Type DATA_BN_5_0[1] = {0.1
};
const D_Type DATA_CONV_5_1[1] = {0.1
};
const D_Type DATA_BN_5_1[1] = {0.1
};

//block 6
const D_Type DATA_CONV_6_0[1] = {0.1
};
const D_Type DATA_BN_6_0[1] = {0.1
};
const D_Type DATA_CONV_6_1[1] = {0.1
};
const D_Type DATA_BN_6_1[1] = {0.1
};

//block 7
const D_Type DATA_CONV_7_0[1] = {0.1
};
const D_Type DATA_BN_7_0[1] = {0.1
};
const D_Type DATA_CONV_7_1[1] = {0.1
};
const D_Type DATA_BN_7_1[1] = {0.1
};
const D_Type DATA_CONV_7_2[1] = {0.1
};
const D_Type DATA_BN_7_2[1] = {0.1
};

//block 8
const D_Type DATA_CONV_8_0[1] = {0.1
};
const D_Type DATA_BN_8_0[1] = {0.1
};
const D_Type DATA_CONV_8_1[1] = {0.1
};
const D_Type DATA_BN_8_1[1] = {0.1
};

//block 9
const D_Type DATA_CONV_9_0[1] = {0.1
};
const D_Type DATA_BN_9_0[1] = {0.1
};
const D_Type DATA_CONV_9_1[1] = {0.1
};
const D_Type DATA_BN_9_1[1] = {0.1
};

//fc
const D_Type DATA_LINEAR_9_0[1] = {0.1
};

const D_Type* data_weight_list[43] = {
    DATA_CONV_0_0,
    DATA_BN_0_0,
//===============
    DATA_CONV_1_0,
    DATA_BN_1_0,
    DATA_CONV_1_1,
    DATA_BN_1_1,

    DATA_CONV_2_0,
    DATA_BN_2_0,
    DATA_CONV_2_1,
    DATA_BN_2_1,

    DATA_CONV_3_0,
    DATA_BN_3_0,
    DATA_CONV_3_1,
    DATA_BN_3_1,
//===============
    DATA_CONV_4_0,
    DATA_BN_4_0,
    DATA_CONV_4_1,
    DATA_BN_4_1,
    DATA_CONV_4_2,
    DATA_BN_4_2,

    DATA_CONV_5_0,
    DATA_BN_5_0,
    DATA_CONV_5_1,
    DATA_BN_5_1,

    DATA_CONV_6_0,
    DATA_BN_6_0,
    DATA_CONV_6_1,
    DATA_BN_6_1,
//==============
    DATA_CONV_7_0,
    DATA_BN_7_0,
    DATA_CONV_7_1,
    DATA_BN_7_1,
    DATA_CONV_7_2,
    DATA_BN_7_2,

    DATA_CONV_8_0,
    DATA_BN_8_0,
    DATA_CONV_8_1,
    DATA_BN_8_1,

    DATA_CONV_9_0,
    DATA_BN_9_0,
    DATA_CONV_9_1,
    DATA_BN_9_1,

    DATA_LINEAR_9_0
};


#endif // _WEIGHT_FILE_H_
