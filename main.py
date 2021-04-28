import sys
import numpy as np
import tensorflow as tf
import keras.backend as K
from environment import Database, Environment
from model import ActorCritic
import argparse
from matplotlib import pyplot as plt

from draw import draw_lines
from configs import parse_cmd_args

if __name__ == "__main__":

    argus = parse_cmd_args()

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
    new_state, reward, done, socre,  _ = env.step(action, 0, 1)  # apply the action -> to steady state -> return the reward
    new_state = new_state.reshape((1, env.state.shape[0]))
    reward_np = np.array([reward])

    print("0-shape")
    print(new_state.shape)
    actor_critic.remember(cur_state, action_2, reward_np, new_state, done)
    actor_critic.train()  # len<32, useless

    cur_state = new_state
    predicted_rewardList = []
    for epoch in range(num_trials):
        # env.render()
        cur_state = cur_state.reshape((1, env.state.shape[0]))
        action, isPredicted = actor_critic.act(cur_state)
        print(action)
        action_2 = action.reshape((1, env.action_space.shape[0]))  # for memory
        # action.tolist()                                          # to execute
        new_state, reward, done, score, _ = env.step(action, isPredicted, epoch + 1)
        new_state = new_state.reshape((1, env.state.shape[0]))
        if isPredicted == 1:
            predicted_rewardList.append([epoch, reward])

        reward_np = np.array([reward])
        print("%d-shape" % epoch)
        print(new_state.shape)

        actor_critic.remember(cur_state, action_2, reward_np, new_state, done)
        actor_critic.train()

        if epoch % 5 == 0:
            actor_critic.actor_model.save_weights('saved_model_weights/actor_weights.h5')
            actor_critic.critic_model.save_weights('saved_model_weights/critic_weights.h5')

        if done or score >argus['stopping_score']:

            print("training end!!")
            break


        cur_state = new_state
    '''
    except:
        print("<>There is an error!<>")
    finally:
        db.close()
    '''