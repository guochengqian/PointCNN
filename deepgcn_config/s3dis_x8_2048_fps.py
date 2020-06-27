#!/usr/bin/python3
import math

num_class = 13

sample_num = 4096

batch_size = 16

num_epochs = 1024

label_weights = [1.0] * num_class

learning_rate_base = 0.001
decay_steps = 5000
decay_rate = 0.5
learning_rate_min = 1e-6
step_val = 500

weight_decay = 1e-8

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, math.pi/32., 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.001, 0.001, 0.001, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 8

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 32 * x, []),
                 (12, 2, 768, 64 * x, []),
                 (16, 2, 384, 96 * x, []),
                 (16, 4, 128, 128 * x, [])]]

with_global = True

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 4, 3, 3),
                  (16, 2, 2, 2),
                  (12, 2, 2, 1),
                  (8, 2, 1, 0)]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(32 * x, 0.0),
              (32 * x, 0.5)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-5

data_dim = 6
use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None

keep_remainder = True

# ======== Parameters for DeepGCN
num_layers = 28  # GCN_layers number [default: 28]
bn_init_decay = 0.5     # BN decay rate for bn decay [default: 0.5]
bn_decay_decay_rate = 0.5
bn_decay_decay_step = 5000
bn_decay_clip = 0.99
num_neighbors = [16]  # The number of k nearest neighbors [Default: 16].
# You can either pass a single value for all layers or one value per layer.
num_filters = [64]    # The number of filers in gcn layers [default: 64]
dilations = [-1]    # The dilation rate for each layer [default: -1 => dilation rate = layer number]

stochastic_dilation = True
sto_dilated_epsilon = 0.2   # Stochastic probability of dilatioin [Default: 0.2]
skip_connect = 'residual'   # Skip Connections (residual, dense, none) [default: residual]
edge_lay = 'dilated'    # The type of edge layers (dilated, knn) [default: dilated]
gcn = 'mrgcn'    # The type of GCN layers (mrgcn, edgeconv) [default: edgeconv]
