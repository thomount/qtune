import sys
import numpy as np
import tensorflow as tf
import keras.backend as K
from environment import Database, Environment
from model import ActorCritic
import argparse
from matplotlib import pyplot as plt


argus = dict()
#arg_key = ['num_trial', 'cur_op', 'num_event', 'p_r_range', 'p_u_index', 'p_i', 'p_d', 'maxlen_mem',
#           'maxlen_predict_mem', 'learning_rate', 'train_min_size']

def parse_cmd_args():
    global argus
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
    args = parser.parse_args()
    argus = vars(args)

    # deep query encoding
    parser.add_argument('--query_vector_dim', type=int, default=4, help='Query Vector Dimension')
    parser.add_argument('--predictor_output_dim', type=int, default=65, help='System Metric Dimension')
    parser.add_argument('--predictor_epoch', type=int, default=500, help='Training Iterations')


def main():

    parse_cmd_args()

    sess = tf.Session()
    K.set_session(sess)
    db = Database(argus) # connector knobs metric
    env = Environment(db, argus)

    actor_critic = ActorCritic(env, sess, learning_rate=argus['learning_rate'], train_min_size=argus['train_min_size'],
                               size_mem=argus['maxlen_mem'], size_predict_mem=argus['maxlen_predict_mem'])

    num_trials = argus['num_trial']  # ?
    # trial_len  = 500   # ?
    # ntp
    env.preheat()

    # First iteration
    cur_state = env._get_obs()  # np.array      (inner_metric + sql)
    cur_state = cur_state.reshape((1, env.state.shape[0]))
    # action = env.action_space.sample()
    action = env.fetch_action()  # np.array
    action_2 = action.reshape((1, env.action_space.shape[0]))  # for memory
    new_state, reward, done, _ = env.step(action, 0, 1)  # apply the action -> to steady state -> return the reward
    new_state = new_state.reshape((1, env.state.shape[0]))
    reward_np = np.array([reward])

    print("0-shape")
    print(new_state.shape)
    actor_critic.remember(cur_state, action_2, reward_np, new_state, done)
    actor_critic.train()  # len<32, useless

    cur_state = new_state
    rewardList = []
    for epoch in range(num_trials):
        # env.render()
        cur_state = cur_state.reshape((1, env.state.shape[0]))
        action, isPredicted = actor_critic.act(cur_state)
        print(action)
        action_2 = action.reshape((1, env.action_space.shape[0]))  # for memory
        # action.tolist()                                          # to execute
        new_state, reward, done, _ = env.step(action, isPredicted, epoch + 1)
        new_state = new_state.reshape((1, env.state.shape[0]))
        rewardList.append(reward)

        reward_np = np.array([reward])
        print("%d-shape" % epoch)
        print(new_state.shape)

        actor_critic.remember(cur_state, action_2, reward_np, new_state, done)
        actor_critic.train()

        if epoch % 5 == 0:
            actor_critic.actor_model.save_weights('saved_model_weights/actor_weights.h5')
            actor_critic.critic_model.save_weights('saved_model_weights/critic_weights.h5')

        if epoch % 10 == 0:
            print('[reward sequence]', rewardList)
            plt.plot(rewardList, color='red')
            plt.savefig('training-results/training.png')


        cur_state = new_state
    '''
    except:
        print("<>There is an error!<>")
    finally:
        db.close()
    '''

if __name__ == "__main__":
    main()