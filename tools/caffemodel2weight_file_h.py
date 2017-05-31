import caffe
import collections
import math
import numpy as np


class LayerNode(object):
    def __init__(self, net_type, layer_id, layer_name,
                 bottom_names, top_names):
        self.net_type = net_type
        self.layer_id = layer_id
        self.layer_name = layer_name

        self.bottom_names = bottom_names
        self.top_names = top_names
        self.weight = None
        self.blobs_shape = None

        self.top_id = []
        self.params = []
        self.net_type_id = None

    r"""net describe file format:

        convoludion:
        format: net_type, layer_id, top_layer, in_plane, out_plane, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w,  bias_term
        length: 3+9

        batch normalization:
        format: net_type, layer_id, top_layer, in_plane
        length: 3+1

        relu:
        format: net_type, layer_id, top_layer
        length: 3+0

        avg pooling:
        format: net_type, layer_id, top_layer, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w
        length: 3+6

        max pooling:
        format: net_type, layer_id, top_layer, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w
        length: 3+6

        linear:
        format: net_type, layer_id, top_layer, in_plane, out_plane, bias_term
        length: 3+3

        scale:
        format: net_type, layer_id, top_layer, in_plane, bias_term
        length: 3+2

        caddtable:
        format: net_type, layer_id, top_layer
        length: 3+0

        concattable:
        format: net_type, layer_id, top_layer_1, top_layer_2;
        length: 4+0

        data layer:
        format: net_type, layer_id, top_layer, batch_size, channel, height, width
        length: 3+4

        ==============================================================
        define net_type_id
        # define NID_CONVOLUTION 0X01
        # define NID_BATCH_NORMALIZATION 0X02
        # define NID_RELU 0X03
        # define NID_AVG_POOLING 0X04
        # define NID_MAX_POOLING 0X05
        # define NID_SCALE 0X06
        # define NID_LINEAR 0X07
        # define NID_CADDTABLE 0X08
        # define NID_CONCATTABLE 0X09
        # define NID_DATA_LAYER 0X0A
        """

    def setparams(self):
        if self.net_type == "BatchNorm":
            self.params = [self.weight[0].data.shape[0]]
            self.net_type_id = 2
        elif self.net_type == "Scale":
            if len(self.weight) == 1:
                bias_term = 0
            else:
                bias_term = 1
            self.params = [self.weight[0].data.shape[0], bias_term]
            self.net_type_id = 6

        elif self.net_type == "Input":
            self.params = [1, 3, 32, 32]
            self.net_type_id = 10

        elif self.net_type == "Convolution":
            if self.weight[0].data.shape[1] >= 16 and (self.weight[0].data.shape[0] != self.weight[0].data.shape[1]):
                stride = 2
            else:
                stride = 1

            if len(self.weight) >= 2:
                bias_term = 1
            else:
                bias_term = 0
            self.params = [self.weight[0].data.shape[1], self.weight[0].data.shape[0],
                           self.weight[0].data.shape[2], self.weight[0].data.shape[3],
                           math.floor(self.weight[0].data.shape[2] / 2.),
                           math.floor(self.weight[0].data.shape[3] / 2.),
                           stride, stride, bias_term]
            self.net_type_id = 1

        elif self.net_type == "Split":
            self.params = []
            self.net_type_id = 9

        elif self.net_type == "ReLU":
            self.params = []
            self.net_type_id = 3

        elif self.net_type == "Eltwise":
            self.params = []
            self.net_type_id = 8

        elif self.net_type == "Pooling":
            self.params = [8, 8, 0, 0, 1, 1]
            self.net_type_id = 4

        elif self.net_type == "InnerProduct":
            if len(self.weight) >= 2:
                bias_term = 1
            else:
                bias_term = 0
            self.params = [self.weight[0].data.shape[1], self.weight[0].data.shape[0], bias_term]
            self.net_type_id = 7
        else:
            print ">>>do nothing"


def main():
    proto_txt_dir = "model/resNet20_cifar10_deploy.prototxt"
    caffe_model_dir = "model/resNet.caffemodel"

    caffe.set_mode_cpu()
    net = caffe.Net(proto_txt_dir, caffe_model_dir, caffe.TEST)

    layer_id = 0
    node_list = []

    bottom_link_list = collections.OrderedDict()
    top_link_list = collections.OrderedDict()
    name_projection_dict = {}

    replace_flag = False
    for layer_name, bottom_names in net.bottom_names.items():
        node = LayerNode(net_type=net.layers[layer_id].type, layer_id=layer_id, layer_name=layer_name,
                         bottom_names=bottom_names, top_names=net.top_names.get(layer_name))

        node.weight = net.params.get(layer_name)
        if net.blobs.get(layer_name) is not None:
            node.blobs_shape = net.blobs.get(layer_name).data.shape

        if len(node.bottom_names) == 1 and len(node.top_names) == 1 and node.bottom_names[0] == node.top_names[0]:
            name_projection_dict[node.top_names[0]] = layer_name
            node.top_names[0] = layer_name
            node.bottom_names[0] = node_list[layer_id - 1].layer_name
        else:
            for i in range(len(node.bottom_names)):
                if node.bottom_names[i] in name_projection_dict:
                    node.bottom_names[i] = name_projection_dict.get(node.bottom_names[i])

        for bottom_items in node.bottom_names:
            if bottom_items in bottom_link_list:
                bottom_link_list[bottom_items].append(layer_id)
            else:
                bottom_link_list[bottom_items] = [layer_id]

        for top_items in node.top_names:
            if top_items in top_link_list:
                top_link_list[top_items].append(layer_id)
            else:
                top_link_list[top_items] = [layer_id]
        node.setparams()
        node_list.append(node)
        layer_id += 1

    for name, top_id in top_link_list.items():
        print top_id
        if name in bottom_link_list:
            node_list[top_id[0]].top_id.append(bottom_link_list[name][0])

    weight_h_path = "weight_file.h"

    weight_h_f = open(weight_h_path, "w")

    weight_h_f.write('''#ifndef _WEIGHT_FILE_H_\n#define _WEIGHT_FILE_H_\n\n#include "utils.h" ''')
    weight_h_f.write('''/**\nthis file only for saving data of weights
name format: DATA_netType_blockID_layerID, i.e. DATA_CONV_1_1
*/
/**example for resnet-20*/\n''')

    weight_count = 0
    for node_item in node_list:
        if node_item.weight is not None:
            flat_weight = None
            data_len = 0
            for weight_item in node_item.weight:
                data_len += weight_item.data.size
                if flat_weight is None:
                    flat_weight = weight_item.data.reshape(weight_item.data.size, 1)
                else:
                    flat_weight = np.row_stack((flat_weight, weight_item.data.reshape(weight_item.data.size, 1)))
            flat_weight = flat_weight.reshape(flat_weight.size)
            weight_h_f.write("//Net Type:%s Params:" % node_item.net_type)
            for params_item in node_item.params:
                weight_h_f.write("%d, " % params_item)
            weight_h_f.write("\nD_Type %s[%d] = {\n" % (node_item.layer_name, data_len))
            for i in range(data_len):
                if i+1 == data_len:
                    weight_h_f.write("%f " % flat_weight[i])
                else:
                    weight_h_f.write("%f, " % flat_weight[i])
                    if (i+1) % 16 == 0:
                        weight_h_f.write("\n")
            weight_h_f.write("\n};\n")
            weight_count += 1

    weight_h_f.write("D_Type* data_weight_list[%d] = {\n" % weight_count)
    for node_item in node_list:
        if node_item.weight is not None:
            weight_h_f.write("    %s," % node_item.layer_name)
            weight_h_f.write("//Net Type:%s Params:" % node_item.net_type)
            for params_item in node_item.params:
                weight_h_f.write("%d, " % params_item)
            weight_h_f.write("\n")
    weight_h_f.write("};\n#endif\n")

    weight_h_f.close()
    print ">>> Create Head File Done!"

if __name__ == '__main__':
    main()
