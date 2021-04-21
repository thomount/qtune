import numpy as np
import pandas
import json
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

from configs import  query_vector_dim, predictor_output_dim, predictor_epoch, dir

# base prediction model
def baseline_model(num_feature=query_vector_dim):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=num_feature, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(predictor_output_dim, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

# Query2CostVector
def compute_cost(node):
    return float(node["Total Cost"]) - float(node["Startup Cost"])

def extract_plan(sample):
    # function: extract SQL cost feature
    # return: cost vector

    mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3}

    plan = sample["Plan"]
#    while isinstance(plan, list):
#        plan = plan[0]
    # Features: print(plan.keys())
    # start time = plan["start_time"]

#    plan = plan["Plan"]  # root node
    sql_vector = [0] * query_vector_dim

    stack = [plan]
    while stack != []:
        parent = stack.pop(0)

        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)

    maxnm = 0
    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        maxnm = maxnm + 1
        run_cost = compute_cost(parent)
        if parent["Node Type"] in mp_optype:
            op_id = mp_optype[parent["Node Type"]]
            sql_vector[int(op_id)] = sql_vector[int(op_id)] + run_cost

        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)

    return sql_vector


class SqlParser:
    ###########################################################################
    # DML: select delete insert update      0 1 2 3
        # select {select_expr}
        # <modifier> (The first is default)
        # [ALL | DISTINCT | DISTINCTROW]
        # [0 | HIGH_PRIORITY], faster than update, with table-level lock
        # [0 | STRAIGHT_JOIN],
        # [0 | SQL_SMALL_RESULT | SQL_BIG_RESULT]
        # [0 | SQL_BUFFER_RESULT]
        # [SQL_CACHE | SQL_NO_CACHE]
        # [SQL_CALC_FOUND_ROWS]

    # OLTP (workload2vector)
        # select*w1 + sum(modifiers)*w2 + num({select_expr})*wl3        # 0.7 0.1 0.2
        # from [table]
        # [WHERE where_condition]   range join

    # OLTP (sql2vector)
        # cost-vector: [Aggregate, Nested Loop, Index Scan, Hash_Join]

    # Keywords
        # [GROUP BY {col_name | expr | position}]
        # [ASC | DESC], ...[WITH ROLLUP]
        # [HAVING where_condition]
        # [ORDER BY {col_name | expr | position}]
        # [ASC | DESC], ...

    # sum(group_table_scale(having)*wi) + order_cost*wi
    ###########################################################################

    def __init__(self, argus):

        # benchmrking command
        self.benchmark = argus['benchmark']
        if self.benchmark == 'sysbench':
            # self.cmd = "sysbench --test=oltp --oltp-table-size=5000 " + " --num-threads=5 --max-requests=" + str(
            #    num_event) + " --mysql-host=127.0.0.1 --mysql-user='{}' --mysql-password='{}' --mysql-port=3306 --db-ps-mode=disable --mysql-db='test' \
            #          --oltp-simple-ranges=" + str(int(num_event * p_r_range)) + " --oltp-index-updates=" + str(int(num_event * p_u_index))
            # sysbench --test=oltp_read_write --mysql-host=166.111.5.177 --mysql-user=root --mysql-password=123456
            # --mysql-db=test --tables=5 --table-size=500000 prepare

            self.cmd = "sysbench --test={}  --db-driver=mysql --tables={} --table-size={} --threads={} --events={} --mysql-host={}" \
                       " --mysql-user={} --mysql-password={} --mysql-port={} --db-ps-mode=disable --mysql-db='test' ".format(
                argus['cur_op'], str(argus['tables']), str(argus['table_size']),'5',str(argus['num_event']),argus['host'],argus['user'],argus['password'],argus['port'])
                #,str(argus['num_event']*argus['p_r_range']),str(argus['num_event']*argus['p_u_index']))
        elif argus['benchmark'] == 'job':
            self.cmd = dir+"run_job.sh "+dir+"all_queries.txt "+dir+"queries-mysql/ "+dir+"output.res"
            print(self.cmd)
            self.resfile = dir+"output.res"
        self.argus = argus

        # sql encoding: DML, tables, operation costs
        # join: do join among imdb tables
        self.join_cmd = ''

        ########### Convert from the sql statement to the sql vector
        #  directly read vector from a file (so a python2 script needs to run first!)
        #  sql_type * (num_events, C, aggregation, in-mem)
        #############################################################################################################################

        # query encoding features
        self.op_weight = {'oltp_point_select': 1, 'select_random_ranges': 2, 'oltp_delete': 3,
                          'oltp_insert': 4, 'bulk_insert': 5, 'oltp_update_index': 6,
                          'oltp_update_non_index': 7, }
        self.cur_op = argus['cur_op']
        self.num_event = argus['num_event']
        self.C = [10000]
        self.group_cost = 0
        self.in_mem = 0

        # Prepare Data
        fs = open("training-data/trainData_sql.txt", 'r')
        df = pandas.read_csv(fs, sep=' ', header=None)
        lt_sql = df.values
        # seperate into input X and output Y
        sql_op = lt_sql[:, 0]
        sql_X = lt_sql[:, 1:5]  # op_type   events  table_size
        sql_Y = lt_sql[:, 5:]
        print(sql_Y[0])

        for i, s in enumerate(sql_op):
            s = s + 1
            sql_X[i][0] = sql_X[i][0] * s
            sql_X[i][1] = sql_X[i][1] * s
            sql_X[i][2] = sql_X[i][2] * s
            sql_X[i][3] = sql_X[i][3] * s

        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(sql_X)
        # X_test = X_train[50:]
        # X_train = X_train[:50]

        sc_Y = StandardScaler()
        Y_train = sc_Y.fit_transform(sql_Y)
        Y_test = Y_train[50:]
        # Y_train = Y_train[:50]

        # Create the sql convert model
        # evaluate model with standardized dataset
        seed = 7
        np.random.seed(seed)
        # estimators = []
        # estimators.append(('standardize', StandardScaler()))
        # estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=50, verbose=0)))
        print("[Predictor] Training Start")
        self.estimator = KerasRegressor(build_fn=baseline_model, epochs=predictor_epoch, batch_size=50, verbose=1)  # epochs
        self.estimator.fit(X_train, Y_train)
        print("[Predictor] Training Finished")



    def query_encoding(self):

        sql_vector = [0]*query_vector_dim
        if self.benchmark == 'sysbench':

            if self.cur_op == "oltp_read_write":
                op_weight = self.op_weight['select_random_ranges'] * self.argus['p_r_range'] + self.op_weight[
                    'oltp_update_index'] * self.argus['p_u_index'] + self.op_weight['oltp_insert'] * self.argus['p_i'] + self.op_weight[
                                'oltp_delete'] * self.argus['p_d']
                print("op_weight:%f" % op_weight)
            else:
                op_weight = self.op_weight[self.cur_op]

            sql_vector = np.array([[self.num_event, self.C[0], self.group_cost, self.in_mem]])
            sql_vector = np.array([[o * op_weight for o in sql_vector[0]]])

        elif self.benchmark == 'job':

            with open(os.getcwd()+"/scripts/job/sampled_query_plans.txt", "r") as f:
                for sample in f.readlines():
                    plan = json.loads(sample)[0]
                    cost_vector = extract_plan(plan)

                    sql_vector = np.add(sql_vector,cost_vector)
            sql_vector = np.array([sql_vector])
        print("[sql vector] ",sql_vector)

        return sql_vector

    def predict_sql_resource(self):
        # Predict sql convert
        # inner_metric_change   np.array
        return self.estimator.predict(
            self.query_encoding())  # input : np.array([[...]])      (sq_type, num_events, C, aggregation, in-mem)
        # output : np.array([[...]])

    def update(self):
        pass
