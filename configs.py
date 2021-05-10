# -*- coding: utf-8 -*-
"""
desciption: system variables or other constant information
"""
import os
import requests
import argparse

#arg_key = ['num_trial', 'cur_op', 'num_event', 'p_r_range', 'p_u_index', 'p_i', 'p_d', 'maxlen_mem',
#           'maxlen_predict_mem', 'learning_rate', 'train_min_size']
def parse_cmd_args():
    ##########################################################
    #                       Sys options
    # num_trial:500
    # cur_op:['oltp_point_select.lua', 'select_random_ranges.lua', 'oltp_delete.lua', 'oltp_insert.lua', 'bulk_insert.lua', 'oltp_update_index.lua', 'oltp_update_non_index.lua’, ‘oltp_read_write.lua’]
    # request_times:1000
    # p_r_range:0.6
    # p_u_index:0.2
    # p_i:0.1
    # p_d:p_i
    # maxlen_mem:2000
    # maxlen_predict_mem:2000
    # learning_rate:0.001
    # train_min_size:32	(self.batch_size)
    ##########################################################

    parser = argparse.ArgumentParser()
    # database
    parser.add_argument('--host', type=str, default='166.111.5.177', help='Host IP Address')
    parser.add_argument('--port', type=str, default=3306, help='Host Port Number')
    parser.add_argument('--user', type=str, default='root', help='Database User Name')
    parser.add_argument('--password', type=str, default='', help='Database Password')
    parser.add_argument('--hostuser', type=str, default='xuanhe', help='Host User Name')
    parser.add_argument('--hostpassword', type=str, default='db10204', help='Host Password')
    parser.add_argument('--knob_num', type=int, default=12, help='Number of Knobs to be Tuned')
    # benchmark
    parser.add_argument('--benchmark', type=str, default='job', help='Benchmark Type: [job, tpch, sysbench]')
    # benchmark [sysbench]
    parser.add_argument('--cur_op', type=str, default='oltp_read_write', help='Workload Type')
    parser.add_argument('--tables', type=int, default=5, help='Table numbers')
    parser.add_argument('--table_size', type=int, default=500000, help='Table size (B)')
    parser.add_argument('--num_event', type=int, default=1000, help='Limit for the requests (queries) in a workload')
    parser.add_argument('--p_r_range', type=int, default=0.6, help='Read percentage')
    parser.add_argument('--p_u_index', type=int, default=0.2, help='Update percentage')
    parser.add_argument('--p_i', type=int, default=0.1, help='Insert percentage')
    parser.add_argument('--p_d', type=int, default=0.1, help='Delete percentage')

    # reinforcement learning
    parser.add_argument('--num_trial', type=int, default=500, help='Iteration Number')
    parser.add_argument('--maxlen_mem', type=int, default=2000, help='Maximum sample number cached in RL')
    parser.add_argument('--maxlen_predict_mem', type=int, default=2000, help='')
    parser.add_argument('--learning_rate', type=int, default=1e-3, help='')
    parser.add_argument('--train_min_size', type=int, default=3, help='Sample threshold to train RL')

    # deep query encoding
    parser.add_argument('--query_vector_dim', type=int, default=4, help='Query Vector Dimension')
    parser.add_argument('--predictor_output_dim', type=int, default=65, help='System Metric Dimension')
    parser.add_argument('--predictor_epoch', type=int, default=500, help='Training Iterations')

    # stopping condition (50)
    parser.add_argument('--stopping_score', type=int, default=50, help='Training finish if the accumulated score is over the value')

    # draw results linelist
    parser.add_argument('--linelist', type=list, default=['res_predict', 'res_random'],
                        help='Training Performance Comparision')
    parser.add_argument('--performance_metric', type=list, default='Latency',
                        help='[Latency, Throughput]')


    args = parser.parse_args()
    argus = vars(args)

    return argus

    # reward
    # draw lines

    # spark 13

# https://dev.mysql.com/doc/refman/5.7/en/dynamic-system-variables.html
knob_config = {
    # you should increase the value of the table_open_cache variable if the numer of opened tables is large
    'table_open_cache' : {
        'type':'integer',
        'min_value':1,
        'max_value':2000,
    },
    # The maximum permitted number of simultaneous client connections
    'max_connections': {
        'type': 'integer',
        'min_value': 1,
        'max_value': 100000,
    },
    # The size of buffer pool (bytes). A larger buffer pool requires less disk I/O to access the same table data more than once.
    'innodb_buffer_pool_size': {
        'type': 'integer',
        'min_value': 5242880,
        'max_value': pow(2,64)-1,
    },
    # The minimum size of the buffer that is used for plain index scans, range index scans, and joins that do not use indexes and thus perform full table scans.
    'join_buffer_size': {
        'type': 'integer',
        'min_value': 128,
        'max_value': 4294967295,
    },
    # The size of the buffer used for index blocks
    # 'key_buffer_size': {
    #     'type': 'integer',
    #     'min_value': 8,
    #     'max_value': 4294967295,
    # },
    # The size of the buffer that is allocated when preloading indexes.
    'preload_buffer_size': {
        'type': 'integer',
        'min_value': 1024,
        'max_value': 1073741824,
    },
    # Each session that must perform a sort allocates a buffer of this size.
    'sort_buffer_size': {
        'type': 'integer',
        'min_value': 32768,
        'max_value': 4294967295,
    },
    # The size of the cache to hold changes to the binary log during a transaction.
    'binlog_cache_size': {
        'type': 'integer',
        'min_value': 4096,
        'max_value': 4294967295,
    },
    # The cutoff on the size of index values that determines which filesort algorithm to use.
    'max_length_for_sort_data': {
        'type': 'integer',
        'min_value': 4,
        'max_value': 8388608,
    },
    # This variable limits the total number of prepared statements in the server. Setting the value to 0 disables prepared statements.
    'max_prepared_stmt_count': {
        'type': 'integer',
        'min_value': 0,
        'max_value': 1048576,
    },
    # The number of times that any given stored procedure may be called recursively. The default value for this option is 0, which completely disables recursion in stored procedures. The maximum value is 255.
    'max_sp_recursion_depth': {
        'type': 'integer',
        'min_value': 0,
        'max_value': 255,
    },
    # The maximum number of simultaneous connections permitted to any given MySQL user account. A value of 0 (the default) means “no limit.”
    'max_user_connections': {
        'type': 'integer',
        'min_value': 0,
        'max_value': 4294967295,
    },
    # If this value is greater than 1, MyISAM table indexes are created in parallel.
    'myisam_repair_threads': {
        'type': 'integer',
        'min_value': 1,
        'max_value': 4294967295,
    }
}

# sync with main.py

dir = "/home/xuanhe/join-order-benchmark-mysql-main/"

query_vector_dim = 4

predictor_output_dim = 65

predictor_epoch = 100