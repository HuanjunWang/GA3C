import random

import gym
import tensorflow as tf
import numpy as np


class QNetwork:
    def __init__(self, model_name, num_actions, observation_dim, gamma, seed):
        self.model_name = model_name
        self.num_actions = num_actions
        self.observation_dim = observation_dim
        self.gamma = gamma

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            tf.set_random_seed(seed)
            self._create_graph()
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())
            self._create_tensor_board()

    def train(self, state, action, target):
        # print("action", action)
        # index = self.sess.run(self.index, feed_dict={self.ph_action: action})
        # print("index", index)

        loss, _, summary, step = self.sess.run([self.loss, self.train_op, self.summary_op, self.global_step],
                                               feed_dict={self.ph_observation: state, self.ph_target: target,
                                                          self.ph_action: action})

        self.log_writer.add_summary(summary, step)
        return loss

    def train_batch(self, state, action, reward, next_state, done):
        next_q = self.sess.run(self.max_output, feed_dict={self.ph_observation: next_state})

        target = reward + self.gamma * next_q * done
        self.train(state, action, target)

    def predict_q_value(self, x):
        q_values = self.sess.run(self.output, feed_dict={self.ph_observation: x})
        return q_values

    def predict_action(self, x):
        actions = self.sess.run(self.action, feed_dict={self.ph_observation: x})
        return actions

    def save(self, episode):
        pass

    def load(self):
        pass

    def _create_graph(self):
        self.ph_observation = tf.placeholder(tf.float32, shape=[None, self.observation_dim], name="Observation")
        self.ph_target = tf.placeholder(tf.float32, shape=[None], name="TDTarget")
        self.ph_action = tf.placeholder(tf.int32, shape=[None], name="Action")
        self.global_step = tf.Variable(0, trainable=False, name='step')

        with tf.variable_scope("NN"):
            self.l1 = tf.contrib.layers.fully_connected(inputs=self.ph_observation,
                                                        num_outputs=128,
                                                        activation_fn=tf.nn.relu)
            self.l2 = tf.contrib.layers.fully_connected(inputs=self.l1,
                                                        num_outputs=128,
                                                        activation_fn=tf.nn.relu)
            self.l3 = tf.contrib.layers.fully_connected(inputs=self.l2,
                                                        num_outputs=128,
                                                        activation_fn=tf.nn.relu)

            self.output = tf.contrib.layers.fully_connected(inputs=self.l3,
                                                            num_outputs=self.num_actions,
                                                            activation_fn=None)

        self.max_output = tf.reduce_max(self.output, axis=1, name="MaxQValue")
        self.action = tf.argmax(self.output, axis=1, name="PredictAction")

        with tf.name_scope("Loss"):
            self.N = tf.shape(self.ph_action)[0]
            self.index = tf.stack([tf.range(self.N, dtype=tf.int32), self.ph_action], axis=1)
            self.yy = tf.gather_nd(self.output, self.index)
            self.loss = tf.losses.mean_squared_error(self.ph_target, self.yy)

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Loss", self.loss))
        # for var in tf.trainable_variables():
        #     summaries.append(tf.summary.histogram("weights_%s" % var.name, var))
        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("/tmp/q_learning/%s" % self.model_name, self.graph)


def q_learning(env_name, iter, render, gamma=.98, seed=0):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    model = QNetwork(env_name, env.action_space.n, env.observation_space.shape[0], gamma, seed)

    m_states = []
    m_actions = []
    m_rewards = []
    m_next_states = []
    m_done = []
    total_reward = 0
    for i in range(iter):
        observation = env.reset()
        epoch_reward = 0
        while True:
            if render:
                env.render()

            if np.random.rand() < (.4 - 1e-3 * i):
                action = np.random.randint(0, env.action_space.n)
            else:
                action = model.predict_action([observation])[0]

            new_observation, reward, done, _ = env.step(action)
            m_states.append(observation)
            m_actions.append(action)
            m_rewards.append(reward)
            m_next_states.append(new_observation)
            m_done.append(done)

            total_reward += reward
            epoch_reward += reward
            batch_size = min(len(m_states), 128)
            indexs = random.sample(list(range(len(m_states))), batch_size)
            # model.train_batch([observation], [action], [reward], [new_observation], [done])
            model.train_batch(np.array(m_states)[indexs],
                              np.array(m_actions)[indexs],
                              np.array(m_rewards)[indexs],
                              np.array(m_next_states)[indexs],
                              np.array(m_done)[indexs])
            observation = new_observation

            if done:
                break
        print("epoch_reward:", epoch_reward)

    print("Average Reward: %d" % (total_reward / iter))
    return (total_reward / iter)


if __name__ == '__main__':
    iter = 1000
    gamma = 1.
    render = False
    env_name = 'CartPole-v1'
    result = {}

    seed = 22
    reward = q_learning(env_name=env_name, iter=iter, render=render, gamma=gamma, seed=seed)
    result[seed] = reward

    print(result)
