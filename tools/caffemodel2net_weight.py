import caffe
import collections
import math
import numpy as np


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

    replace_flag = False
    for layer_name, bottom_names in net.bottom_names.items():
        node = LayerNode(net_type=net.layers[layer_id].type, layer_id=layer_id,layer_name=layer_name,
                         bottom_names=bottom_names, top_names=net.top_names.get(layer_name))

        node.weight = net.params.get(layer_name)
        if net.blobs.get(layer_name) is not None:
            node.blobs_shape = net.blobs.get(layer_name).data.shape

        if len(node.bottom_names) == 1 and len(node.top_names) == 1 and node.bottom_names[0] == node.top_names[0]:
            node.top_names[0] = layer_name
            if replace_flag:
                node.bottom_names[0] = node_list[layer_id-1].layer_name
            else:
                replace_flag = True
        else:
            if replace_flag:
                node.bottom_names[0] = node_list[layer_id - 1].layer_name
                replace_flag = False

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
        if name in bottom_link_list:
            node_list[top_id[0]].top_id.append(bottom_link_list[name][0])

    net_dat_path = "network.dat"
    weight_dat_path = "weight.dat"

    net_dat_f = open(net_dat_path, "w")
    weight_dat_f = open(weight_dat_path, "w")

    for node_item in node_list:
        net_dat_f.write("%d %d " % (node_item.net_type_id, node_item.layer_id))
        if len(node_item.top_id) >= 1:
            for top_id_item in node_item.top_id:
                net_dat_f.write("%d " % top_id_item)
        else:
            net_dat_f.write("%d " % (2*len(node_list)))

        for params_item in node_item.params:
            net_dat_f.write("%d " % params_item)
        # net_dat_f.write("\n")

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
            weight_dat_f.write("%d " % node_item.layer_id)
            weight_dat_f.write("%d " % data_len)
            for i in range(data_len):
                weight_dat_f.write("%f " % flat_weight[i])

    net_dat_f.close()
    weight_dat_f.close()
'''
    for node_item in node_list:
        print node_item.net_type, node_item.layer_name, node_item.layer_id, node_item.blobs_shape
        if node_item.weight is not None:
            for weight_items in node_item.weight:
                print weight_items.data.shape'''

if __name__ == '__main__':
    main()
