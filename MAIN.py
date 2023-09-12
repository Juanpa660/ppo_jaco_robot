#! /usr/bin/env python3
import gym
import numpy as np
from gym import wrappers
from POLICYPI import Policy
from V_F import NNValueFunction
import scipy.signal
from UTIL import Scaler
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_episode(env, policy, scaler, animate):
    """ CORRE UN EPISODIO DENTRO DE RUN POLICY Y OBTIENE: REWARDS, OBS Y DONE """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs)
        action_nn = np.append(action, np.array([[0, 0, 0]]))
        actions.append(action.reshape((1, -1)).astype(np.float64))
        obs, reward, done, _ = env.step(action_nn)
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 4e-4  # increment time step feature 1e-3

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, episodes, animate):
    """ CORRE LA POLICY Y AGREGA A TRAJECTORIES: OBS, ACTIONS, REWARDS Y OBSERVACIONES SIN SCALER"""
    path_info = []
    mean_reward = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler, animate)
        path = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        mean_reward.append(np.sum(rewards))
        path_info.append(path)
    unscaled = np.concatenate([t['unscaled_obs'] for t in path_info])
    scaler.update(unscaled)  # update running statistics for scaling observations

    return path_info, np.mean(mean_reward)


def discount(x, gamma):
    """ CALCULA LA RECOMPENSA DESCONTADA """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(path_info, gamma):
    """ AGREGA LA SUMA DE LAS RECOMPENSAS DESCONTADA EN EL TIEMPO POR GAMMA """
    for path in path_info:
        rewards = path['rewards'] * (1 - gamma)
        disc_sum_rew = discount(rewards, gamma)
        path['disc_sum_rew'] = disc_sum_rew


def add_value(path_info, val_func):
    """ AGREGA A TRAJECTORIES VF APROXIMADO PARA CADA OBS, POR LA RED NEURONAL """
    for path in path_info:
        observes = path['observes']
        values = val_func.predict(observes)
        path['values'] = values


def add_gae(path_info, gamma, lam):
    """ AGREGA A TRAJECTORIES GENERAL ADVANTAGE ESTIMATOR """
    for path in path_info:
        rewards = path['rewards'] * (1 - gamma)
        values = path['values']
        delta = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(delta, gamma * lam)
        path['advantages'] = advantages


def build_train_set(trajectories):
    """ ENTREGA LOS DATOS DENTRO DE TRAJECTORIES PARA EL TRAIN SET """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return observes, actions, advantages, disc_sum_rew



def main(env_name, num_episodes, gamma, lam, epsilon, mult_neuronas, batch_size, log_name, policy_std):
    env = gym.make(env_name)
    env.seed(2)
    obs_dim = env.observation_space.shape[0] + 1
    act_dim = env.action_space.shape[0] - 3
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, epsilon, mult_neuronas, 10)
    policy = Policy(obs_dim, act_dim, epsilon, mult_neuronas, 20, policy_std)
    show = False
    n_epi_record = 1000
    writer = tf.summary.FileWriter('./log'+str(log_name), tf.get_default_session())
    env = gym.wrappers.Monitor(env, './log/videos'+str(log_name),
    video_callable=lambda episode_id: episode_id % n_epi_record == 0 or episode_id == num_episodes - 1, force=True)
    run_policy(env, policy, scaler, episodes=5, animate=show)
    episode = 0
    while episode < num_episodes:
        path_info, mean_batch_rewards = run_policy(env, policy, scaler, episodes=batch_size, animate=show)
        episode += len(path_info)

        # CALCULA Y AGREGA A TRAJECTORIES: V, DISC_SUM_REW Y ADV
        add_value(path_info, val_func)
        add_disc_sum_rew(path_info, gamma)
        add_gae(path_info, gamma, lam)

        #  OBTIENE INFO DE TRAJECTORIES
        observes, actions, advantages, disc_sum_rew = build_train_set(path_info)

        # ACTUALIZA POLICY Y VALUE FUNCTION
        #summary = val_func.get_summary(obs=observes, val=disc_sum_rew)
        summary2 = policy.update(observes, actions, advantages)
        val_func.fit(observes, disc_sum_rew)


        # MUESTRA PORCENTAJE Y RECOMPENSA PROMEDIO DE EPISODIO
        if (episode/num_episodes)*100 % 2 == 0:
            print(int(np.round((episode/num_episodes)*100)), '%', 'Episode mean reward:', mean_batch_rewards)


        # INFO PARA TENSORBOARD
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Episode mean reward',simple_value=mean_batch_rewards)]), episode)
        for i in range(len(actions[0])):
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=('Actions/joint'+str(i)), simple_value=actions[0][i])]),episode)
        #writer.add_summary(summary, episode)
        writer.add_summary(summary2, episode)

    writer.close()
    policy.close_sess()
    val_func.close_sess()

if __name__ == "__main__":
    print('MAIN')
    #main(env_name='', num_episodes=, gamma=0.995, lam=0.98, kl_targ=0.003, batch_size=20)
    main(env_name='Jaco-v1', num_episodes=50000, gamma=0.99, lam=0.95, epsilon=0.2, mult_neuronas=20, batch_size=20,
    log_name='/Jaco/m1', policy_std=0)
