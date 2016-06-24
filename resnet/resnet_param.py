import utils

import cPickle as pickle

class ResNetParam():

    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as fd:
            self.model_dict = pickle.load(fd)


    # It's unfortunate that it needs all these parameters but due
    # to the bug mentioned below we have to special case the creation of
    # the kernel.
    def conv_kernel(self, name, in_chans, out_chans, shape, strides, trainable=True):
        k = self.model_dict[name]
        kernel = utils.tf_variable_with_value_weight_decay("kernel", k, trainable)
        return kernel

    def bn_params(self, bn_name, scale_name, depth, trainable=True):
        mean, var, gamma, beta = self.model_dict[bn_name]

        #mean = utils.tf_variable_with_value('mean', mean, trainable)
        #var = utils.tf_variable_with_value('var', var, trainable)
        gamma = utils.tf_variable_with_value('gamma', gamma, trainable)
        beta = utils.tf_variable_with_value('beta', beta, trainable)

        #return mean, var, gamma, beta
        return gamma, beta

    def fc_params(self, name, trainable=True):
        weights, bias = self.model_dict[name]

        w = utils.tf_variable_with_value_weight_decay("weights", weights, trainable)
        b = utils.tf_variable_with_value("bias", bias, trainable)

        return w, b
