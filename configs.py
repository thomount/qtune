# -*- coding: utf-8 -*-
"""
desciption: system variables or other constant information
"""
import os
import requests

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