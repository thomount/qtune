import datetime
import subprocess
from collections import deque
import numpy as np

import pymysql
import pymysql.cursors as pycursor

import gym
from gym import spaces
from gym.utils import seeding
import paramiko

from sql2resource import SqlParser

from configs import knob_config

# fetch all the knobs from the prepared configuration info


class Database:
    def __init__(self, argus):
        self.connection = pymysql.connect(host=argus['host'],
                                          port=argus['port'],
                                          user=argus['user'],
                                          password=argus['password'],
                                          db='INFORMATION_SCHEMA',
                                          cursorclass=pycursor.DictCursor)

        # self.internal_metric_num = 13 # 13(state) + cumulative()
        self.external_metric_num = 2  # [throughput, latency]           # num_event / t

        ###################################### [MYSQL Parameters] ######################################
        # bulk_insert_buffer_size connect_timeout default_week_format delayed_insert_limit  delayed_insert_timeout delayed_queue_size div_precision_increment flush_time
        # host_cache_size interactive_timeout key_buffer_size key_cache_age_threshold  key_cache_block_size key_cache_division_limit
        # log_throttle_queries_not_using_indexes log_warnings
        # max_allowed_packet max_connect_errors max_error_count max_heap_table_size max_insert_delayed_threads(no range)  max_join_size(1, 18446744073709551615)
        # max_length_for_sort_data（4， 8388608） max_prepared_stmt_count（0， 104856） max_seeks_for_key（1， 4294967295）
        # max_sp_recursion_depth（0,255） max_user_connections（0，4294967295） max_write_lock_count（1,，4294967295） min_examined_row_limit（0，4294967295）
        # myisam_max_sort_file_size（？？） myisam_repair_threads（1，4294967295）

        # dynamic parameters, ordered by the performance importance
        # https://dev.mysql.com/doc/refman/5.7/en/dynamic-system-variables.html
        self.knob_num = argus['knob_num']
        if self.knob_num > len(knob_config):
            self.knob_num = len(knob_config)

        self.knob_names = [knob[0] for knob in list(knob_config.items())[:self.knob_num]]

        '''
        system metrics
        self.i_names = ["lock_row_lock_time_max","lock_row_lock_time_avg","buffer_pool_size","buffer_pool_pages_total","buffer_pool_pages_misc",
                        "buffer_pool_pages_data","buffer_pool_bytes_data","buffer_pool_pages_dirty","buffer_pool_bytes_dirty","buffer_pool_pages_free",
                        "trx_rseg_history_len","file_num_open_files","innodb_page_size",]
        ["lock_deadlocks","lock_timeouts","lock_row_lock_current_waits","lock_row_lock_time","" ]
        '''
        self.internal_metric_num = 65  # default system metrics enabled in metric_innodb
        with self.connection.cursor() as cursor:
            sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
            cursor.execute(sql)
            result = cursor.fetchall()
            self.internal_metric_num = len(result)

        # Resource Distribution
        # table (col(value_distr, type, index))

    def close(self):
        self.connection.close()

    def fetch_internal_metrics(self):
        with self.connection.cursor() as cursor:
            ######### observation_space
            #         State_status
            # [lock_row_lock_time_max, lock_row_lock_time_avg, buffer_pool_size,
            # buffer_pool_pages_total, buffer_pool_pages_misc, buffer_pool_pages_data, buffer_pool_bytes_data,
            # buffer_pool_pages_dirty, buffer_pool_bytes_dirty, buffer_pool_pages_free, trx_rseg_history_len,
            # file_num_open_files, innodb_page_size]
            #         Cumulative_status
            # [lock_row_lock_current_waits, ]
            '''
            sql = "select count from INNODB_METRICS where name='lock_row_lock_time_max' or name='lock_row_lock_time_avg'\
            or name='buffer_pool_size' or name='buffer_pool_pages_total' or name='buffer_pool_pages_misc' or name='buffer_pool_pages_data'\
            or name='buffer_pool_bytes_data' or name='buffer_pool_pages_dirty' or name='buffer_pool_bytes_dirty' or name='buffer_pool_pages_free'\
            or name='trx_rseg_history_len' or name='file_num_open_files' or name='innodb_page_size'"
            '''
            sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
            cursor.execute(sql)
            result = cursor.fetchall()
            state_list = np.array([])
            for s in result:
                state_list = np.append(state_list, [s['count']])

            return state_list

    def fetch_knob(self):
        with self.connection.cursor() as cursor:
            ######### action_space
            #         Common part
            # '''
            # sql = "select @@table_open_cache, @@max_connections, @@innodb_buffer_pool_size, @@innodb_buffer_pool_instances,\
            # @@innodb_log_files_in_group, @@innodb_log_file_size, @@innodb_purge_threads, @@innodb_read_io_threads,\
            # @@innodb_write_io_threads, @@binlog_cache_size"
            # '''

            #         Read-only
            # innodb_buffer_pool_instances innodb_log_files_in_group innodb_log_file_size innodb_purge_threads innodb_read_io_threads
            # innodb_write_io_threads

            #         Extended part
            # innodb_adaptive_max_sleep_delay(0,1000000)    innodb_change_buffer_max_size(0,50) innodb_flush_log_at_timeout(1,2700) innodb_flushing_avg_loops(1,1000)
            # innodb_max_purge_lag(0,4294967295)    innodb_old_blocks_pct(5,95) innodb_read_ahead_threshold(0,64)   innodb_replication_delay(0,4294967295)
            # innodb_rollback_segments(1,128)   innodb_adaptive_flushing_lwm(0,70)   innodb_sync_spin_loops (0,4294967295)
            # innodb_lock_wait_timeout(1,1073741824)    innodb_autoextend_increment(1,1000) innodb_concurrency_tickets(1,4294967295)    innodb_max_dirty_pages_pct(0,99)
            # innodb_max_dirty_pages_pct_lwm(0,99)  innodb_io_capacity(100, 2**32-1)    innodb_lru_scan_depth(100, 2**32-1) innodb_old_blocks_time(0, 2**32-1)
            # innodb_purge_batch_size(1,5000)   innodb_spin_wait_delay(0,2**32-1)

            #        Non-dynmic
            # innodb_sync_array_size    metadata_locks_cache_size   metadata_locks_hash_instances   innodb_log_buffer_size  eq_range_index_dive_limit   max_length_for_sort_data
            # read_rnd_buffer_size  table_open_cache_instances  transaction_prealloc_size   binlog_order_commits    query_cache_limit   query_cache_size    query_cache_type    query_prealloc_size
            # join_buffer_size  tmp_table_size  max_seeks_for_key   query_alloc_block_size  sort_buffer_size    thread_cache_size   max_write_lock_count

            sql = "select"
            for i,knob in enumerate(self.knob_names):
                sql = sql + ' @@' + knob
                if i < self.knob_num-1:
                    sql = sql + ', '

            cursor.execute(sql)
            result = cursor.fetchall()
            state_list = np.array([])

            i = 0
            state_list = []
            for i in range(self.knob_num):
                state_list = np.append(state_list, result[0]["@@%s" % self.knob_names[i]])

            return state_list

    def change_knob_nonrestart(self, actions):
        with self.connection.cursor() as cursor:
            for i in range(self.knob_num):
                sql = 'set global %s=%d' % (self.knob_names[i], actions[i])
                cursor.execute(sql)
                # result = cursor.fetchall()
            # connection.commit()
            return 1

# Define the environment
class Environment(gym.Env):

    def __init__(self, db, argus):

        self.db = db

        self.parser = SqlParser(argus)

        self.state_num = db.internal_metric_num
        self.action_num = db.knob_num
        self.benchmark = argus['benchmark']
        self.score = 0

        ''' observation dim = system metric dim + query vector dim '''
        self.o_dim = db.internal_metric_num + len(self.parser.predict_sql_resource()[0])
        # print(db.internal_metric_num, self.parser.predict_sql_resource())
        self.o_low = np.array([-1e+10]*self.o_dim)
        self.o_high = np.array([1e+10]*self.o_dim)

        self.observation_space = spaces.Box(low=self.o_low, high=self.o_high, dtype=np.float32)
        # part 1: current system metric
        self.state = db.fetch_internal_metrics()
        # print("Concatenated state:")
        # part 2: predicted system metric after executing the workload
        self.state = np.append(self.parser.predict_sql_resource(), self.state)
        print(self.state)

        ''' action space '''
        # Offline
            # table_open_cache(1), max_connections(2), innodb_buffer_pool_instances(4),
            # innodb_log_files_in_group(5), innodb_log_file_size(6), innodb_purge_threads(7), innodb_read_io_threads(8)
            # innodb_write_io_threads(9),
        # Online
            # innodb_buffer_pool_size(3), max_binlog_cache_size(10), binlog_cache_size(11)
            # 1 2 3 11
            # exclude
            # innodb_file_per_table, skip_name_resolve, binlog_checksum,
            # binlog_format(dynamic, [ROW, STATEMENT, MIXED]),

        self.a_low = np.array([knob[1]['min_value'] for knob in list(knob_config.items())[:db.knob_num]])
        self.a_high = np.array([knob[1]['max_value'] for knob in list(knob_config.items())[:db.knob_num]])

        self.action_space = spaces.Box(low=self.a_low, high=self.a_high, dtype=np.float32)
        self.default_action = self.a_low

        self.mem = deque(maxlen=argus['maxlen_mem'])  # [throughput, latency]
        self.predicted_mem = deque(maxlen=argus['maxlen_predict_mem'])

        self.seed()
        self.start_time = datetime.datetime.now()

        # initialize the SSH client
        self.client = paramiko.SSHClient()
        # add to known hosts
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.client.connect(hostname=argus['host'], username=argus['hostuser'], password=argus['hostpassword'])
        except:
            print("[!] Cannot connect to the SSH Server")
            exit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def execute_sysbench(self, command):
        stdin, stdout, stderr = self.client.exec_command(command)
        result = stdout.read().decode()

        return result

    def execute_job(self, command):
        stdin, stdout, stderr = self.client.exec_command(command)
        result = stdout.read().decode()

        return result

    def preheat(self):
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_read_only.lua --threads=4 --events=0 --time=40 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph' --mysql-port=3306 --tables=5 --table-size=1000000 --range_selects=off --db-ps-mode=disable --report-interval=1 --mysql-db='sbtest' run >/home/zxh/fl_preheat 2>&1"
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_write_only.lua --threads=10 --time=40 --events=0 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph'\
        # --mysql-port=3306 --tables=10 --table-size=500000 --db-ps-mode=disable --report-interval=10 --mysql-db='sbtest_wo_2' run >/home/zxh/fl_preheat 2>&1"
        # p = subprocess.check_output(self.parser.cmd + ' cleanup', shell=True)

        if self.benchmark == 'sysbench':
            p = self.execute_sysbench(self.parser.cmd + ' prepare')
            p = self.execute_sysbench(self.parser.cmd + ' run')
            p = self.execute_sysbench(self.parser.cmd + ' cleanup')
        elif self.benchmark == 'job':
            p = self.execute_job(self.parser.cmd)
        '''
        p = subprocess.check_output(self.parser.cmd + ' prepare', shell=True)

        p = subprocess.check_output(self.parser.cmd + ' run', shell=True)

        p = subprocess.check_output(self.parser.cmd + ' cleanup', shell=True)        
        '''
        print("[Preheat] Finished")

    def fetch_action(self):
        return self.db.fetch_knob()

    # new_state, reward, done,
    def step(self, u, isPredicted, iteration):
        flag = self.db.change_knob_nonrestart(u)

        # if failing to tune knobs, give a high panlty
        if not flag:
            return self.state, -10e+4, False, self.score, {}

        # 1 run sysbench
        # primary key lookup
        # cmd = "sysbench oltp_read_write.lua --threads=4 --events=0 --time=20 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph' --mysql-port=3306 --tables=5 --table-size=1000000 --range_selects=off --db-ps-mode=disable --report-interval=1 --mysql-db='sbtest' run >/home/zxh/fl1 2>&1"
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_write_only.lua --threads=10 --time=30 --events=0 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph'\
        # --mysql-port=3306 --tables=10 --table-size=500000 --db-ps-mode=disable --report-interval=10 --mysql-db='sbtest_wo_2' run >/home/zxh/fl1 2>&1"
        # self.parser.cmd
        # p = subprocess.check_output(self.parser.cmd + ' cleanup', shell=True)

        if self.benchmark == 'sysbench':
            p = self.execute_sysbench(self.parser.cmd + ' prepare')
            p = self.execute_sysbench(self.parser.cmd + ' run >fl1 2>&1')
            p = self.execute_sysbench(self.parser.cmd + ' cleanup')
            ifs = self.execute_sysbench("cat fl1")
            sftp_client = self.client.open_sftp()
            remote_file = sftp_client.open('fl1')
            try:
                for line in remote_file.readlines():
                    a = line.split()
                    if len(a) > 2 and 'transactions:' == a[0]:
                        throughput = float(a[2][1:])
                        # print('T: '+str(throughput))
                    if len(a) > 1 and 'avg:' == a[0]:
                        latency = float(a[1][:2])
            # process line
            finally:
                remote_file.close()
        elif self.benchmark == 'job':
            p = self.execute_job(self.parser.cmd)
            # ifs = self.execute_tpch(self.parser.output)
            sftp_client = self.client.open_sftp()
            remote_file = sftp_client.open(self.parser.resfile)
            try:
                for line in remote_file.readlines():
                    a = line.split()
                    if len(a) > 1 and 'avg_tps(txn/min):' == a[0]:
                        throughput = float(a[1])
                        # print('T: '+str(throughput))
                    if len(a) > 1 and 'avg_lat(ms):' == a[0]:
                        latency = float(a[1])
            # process line
            finally:
                remote_file.close()
        else:
            print("no benchmark is matched: {}".format(self.benchmark))
            exit()

        #ifs = open('fl1', 'r')
        # print(str(len(self.mem)+1)+"\t"+str(throughput)+"\t"+str(latency))
        cur_time = datetime.datetime.now()
        interval = (cur_time - self.start_time).seconds
        self.mem.append([throughput, latency])
        # 2 refetch state
        self._get_obs()

        # 3 cul reward(T, L)
        if len(self.mem) != 0:
            dt0 = (throughput - self.mem[0][0]) / self.mem[0][0]
            dt1 = (throughput - self.mem[len(self.mem) - 1][0]) / self.mem[len(self.mem) - 1][0]
            if dt0 >= 0:
                rt = ((1 + dt0) ** 2 - 1) * abs(1 + dt1)
            else:
                rt = -((1 - dt0) ** 2 - 1) * abs(1 - dt1)

            dl0 = -(latency - self.mem[0][1]) / self.mem[0][1]

            dl1 = -(latency - self.mem[len(self.mem) - 1][1]) / self.mem[len(self.mem) - 1][1]

            if dl0 >= 0:
                rl = ((1 + dl0) ** 2 - 1) * abs(1 + dl1)
            else:
                rl = -((1 - dl0) ** 2 - 1) * abs(1 - dl1)

        else:  # initial action
            rt = 0
            rl = 0

        reward = 6 * rl + 4 * rt

        '''
        reward = 0
        for i in range(u.shape[0]):
            tmp = u[i] / self.a_high[i]
            reward+=tmp
        print("Performance: %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))
        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])
            if len(self.predicted_mem)%10 == 0:
                print("Predict List")
                print(self.predicted_mem)
       '''

        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])
            # if len(self.predicted_mem)%10 == 0:
            # print("Predict List")
            # print(self.predicted_mem)
            print("Predict %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            self.pfs = open('training-results/rw_predict_2', 'a')
            self.pfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            self.pfs.close()

            self.score = self.score + reward

        else:
            print("Random %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            self.rfs = open('training-results/rw_random_2', 'a')
            self.rfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            self.rfs.close()

        return self.state, reward, False, self.score, {}

    def _get_obs(self):
        self.state = self.db.fetch_internal_metrics()
        self.state = np.append(self.parser.predict_sql_resource(), self.state)
        return self.state
