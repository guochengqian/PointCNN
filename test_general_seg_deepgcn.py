#!/usr/bin/python3
"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import h5py
import argparse
import importlib
import data_utils
import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import partial, update_wrapper
import tf_util
from gcn_lib import tf_vertex, tf_edge, tf_nn
from gcn_lib.gcn_utils import VertexLayer, EdgeLayer


# todo: change the saving dir.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-t', help='Path to input .h5 filelist (.txt)', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--max_point_num', '-p', help='Max point number of each sample', type=int, default=8192)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=4)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--save_ply', '-s', help='Save results as ply', action='store_true')
    args = parser.parse_args()
    print(args)

    # model = importlib.import_module(args.model)
    model_builder = __import__(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model + '_config')
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    sample_num = setting.sample_num
    max_point_num = args.max_point_num
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    # parametes for deepgcn
    BN_INIT_DECAY = setting.bn_init_decay
    BN_DECAY_DECAY_RATE = setting.bn_decay_decay_rate
    BN_DECAY_DECAY_STEP = setting.bn_decay_decay_step
    BN_DECAY_CLIP = setting.bn_decay_clip

    # GCN parameters
    NUM_LAYERS = setting.num_layers
    NUM_NEIGHBORS = setting.num_neighbors
    if (len(NUM_NEIGHBORS) < NUM_LAYERS):
        while (len(NUM_NEIGHBORS) < NUM_LAYERS):
            NUM_NEIGHBORS.append(NUM_NEIGHBORS[-1])

    NUM_FILTERS = setting.num_filters
    if (len(NUM_FILTERS) < NUM_LAYERS):
        while (len(NUM_FILTERS) < NUM_LAYERS):
            NUM_FILTERS.append(NUM_FILTERS[-1])

    DILATIONS = setting.dilations
    if DILATIONS[0] < 0:
        DILATIONS = [1] + list(range(1, NUM_LAYERS))
    elif (len(DILATIONS) < NUM_LAYERS):
        while (len(DILATIONS) < NUM_LAYERS):
            DILATIONS.extend(DILATIONS)
        while (len(DILATIONS) > NUM_LAYERS):
            DILATIONS.pop()

    STOCHASTIC_DILATION = setting.stochastic_dilation
    STO_DILATED_EPSILON = setting.sto_dilated_epsilon
    SKIP_CONNECT = setting.skip_connect
    EDGE_LAY = setting.edge_lay

    GCN = setting.gcn
    if GCN == "mrgcn":
        print("Using max relative gcn")
    elif GCN == 'edgeconv':
        print("Using edgeconv gcn")
    elif GCN == 'graphsage':
        NORMALIZE_SAGE = setting.normalize_sage
        print("Using graphsage with normalize={}".format(NORMALIZE_SAGE))
    elif GCN == 'gin':
        ZERO_EPSILON_GIN = setting.zero_epsilon_gin
        print("Using gin with zero epsilon={}".format(ZERO_EPSILON_GIN))
    else:
        raise Exception("Unknow gcn")

    def wrapped_partial(func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, sample_num, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='points')
    ######################################################################

    ######################################################################
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if not setting.use_extra_features:
            features_sampled = None
    else:
        points_sampled = pts_fts_sampled
        features_sampled = None

    # DeepGCNs
    batch = tf.Variable(0, trainable=False)
    bn_decay = tf_util.get_bn_decay(batch,
                                    BN_INIT_DECAY,
                                    batch_size,
                                    BN_DECAY_DECAY_STEP,
                                    BN_DECAY_DECAY_RATE,
                                    BN_DECAY_CLIP)
    tf.summary.scalar('bn_decay', bn_decay)

    # Configure the neural network using every layers
    nn = tf_nn.MLP(kernel_size=[1, 1],
                   stride=[1, 1],
                   padding='VALID',
                   weight_decay=0.0,
                   bn=True,
                   bn_decay=bn_decay,
                   is_dist=True)

    # Configure the gcn vertex layer object
    if GCN == 'mrgcn':
        v_layer = tf_vertex.max_relat_conv_layer
    elif GCN == 'edgeconv':
        v_layer = tf_vertex.edge_conv_layer
    else:
        raise Exception("Unknown gcn type")
    v_layer_builder = VertexLayer(v_layer,
                                  nn)

    # Configure the gcn edge layer object
    if EDGE_LAY == 'dilated':
        e_layer = wrapped_partial(tf_edge.dilated_knn_graph,
                                  stochastic=STOCHASTIC_DILATION,
                                  epsilon=STO_DILATED_EPSILON)
    elif EDGE_LAY == 'knn':
        e_layer = tf_edge.knn_graph
    else:
        raise Exception("Unknown edge layer type")
    distance_metric = tf_util.pairwise_distance

    e_layer_builder = EdgeLayer(e_layer,
                                distance_metric)

    # Get the whole model builer
    net = model_builder.Model(tf.concat([points_sampled, features_sampled], -1),
                              is_training,
                              sample_num,
                              NUM_LAYERS,
                              NUM_NEIGHBORS,
                              NUM_FILTERS,
                              setting.num_class,
                              vertex_layer_builder=v_layer_builder,
                              edge_layer_builder=e_layer_builder,
                              mlp_builder=nn,
                              skip_connect=SKIP_CONNECT,
                              dilations=DILATIONS)
    # net = model.Net(points_sampled, features_sampled, is_training, setting)
    seg_probs_op = tf.nn.softmax(net.pred, name='seg_probs')

    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        folder = os.path.dirname(args.filelist)
        filenames = [os.path.join(folder, line.strip()) for line in open(args.filelist)]
        for filename in filenames:
            print('{}-Reading {}...'.format(datetime.now(), filename))
            with h5py.File(filename, "r") as data_h5:
                data = data_h5['data'][...].astype(np.float32)
                data_num = data_h5['data_num'][...].astype(np.int32)
                batch_num = data.shape[0]
                indices_split_to_full = data_h5['indices_split_to_full'][...]

            labels_pred = np.full((batch_num, max_point_num), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, max_point_num), dtype=np.float32)

            print('{}-{:d} testing batches.'.format(datetime.now(), batch_num))
            for batch_idx in range(batch_num):
                if batch_idx % 10 == 0:
                    print('{}-Processing {} of {} batches.'.format(datetime.now(), batch_idx, batch_num))
                points_batch = data[[batch_idx] * batch_size, ...]
                point_num = data_num[batch_idx]

                tile_num = math.ceil((sample_num * batch_size) / point_num)
                indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
                np.random.shuffle(indices_shuffle)
                indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
                indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

                seg_probs = sess.run([seg_probs_op],
                                     feed_dict={
                                         pts_fts: points_batch,
                                         indices: indices_batch,
                                         is_training: False,
                                     })
                probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))

                predictions = [(-1, 0.0)] * point_num
                for idx in range(sample_num * batch_size):
                    point_idx = indices_shuffle[idx]
                    probs = probs_2d[idx, :]
                    confidence = np.amax(probs)
                    label = np.argmax(probs)
                    if confidence > predictions[point_idx][1]:
                        predictions[point_idx] = [label, confidence]
                labels_pred[batch_idx, 0:point_num] = np.array([label for label, _ in predictions])
                confidences_pred[batch_idx, 0:point_num] = np.array([confidence for _, confidence in predictions])

            filename_pred = filename[:-3] + '_pred.h5'
            print('{}-Saving {}...'.format(datetime.now(), filename_pred))
            with h5py.File(filename_pred, 'w') as file:
                file.create_dataset('data_num', data=data_num)
                file.create_dataset('label_seg', data=labels_pred)
                file.create_dataset('confidence', data=confidences_pred)
                file.create_dataset('indices_split_to_full', data=indices_split_to_full)
                file.close()

            if args.save_ply:
                print('{}-Saving ply of {}...'.format(datetime.now(), filename_pred))
                filepath_label_ply = os.path.join(filename_pred[:-3] + 'ply_label')
                data_utils.save_ply_property_batch(data[:, :, 0:3], labels_pred[...],
                                                   filepath_label_ply, data_num[...], setting.num_class)
            ######################################################################
        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
