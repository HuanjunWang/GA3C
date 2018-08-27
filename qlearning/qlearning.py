import random

import gym
import tensorflow as tf
import numpy as np


class QNetwork:
    def __init__(self, model_name, num_actions, observation_dim, gamma, seed, log_dir):
        self.model_name = model_name
        self.num_actions = num_actions
        self.observation_dim = observation_dim
        self.gamma = gamma
        self.log_dir = log_dir

        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(seed)
            self._create_graph()
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())
            self._create_tensor_board()
            tf.keras.backend.set_session(self.sess)


    def train(self, state, action, target):
        loss, _, summary, step = self.sess.run([self.loss, self.train_op, self.summary_op, self.global_step],
                                               feed_dict={self.ph_observation: state, self.ph_target: target,
                                                          self.ph_action: action})

        self._copy_to_target_nn()

        self.log_writer.add_summary(summary, step)
        return loss

    def train_batch(self, state, action, reward, next_state, done):
        next_q = self.sess.run(self.max_output, feed_dict={self.ph_observation: next_state})
        not_done = ~done
        target = reward + self.gamma * next_q * not_done
        self.train(state, action, target)

    def predict_q_value(self, x):
        q_values = self.sess.run(self.output, feed_dict={self.ph_observation: x})
        return q_values

    def predict_action(self, x):
        actions = self.sess.run(self.action, feed_dict={self.ph_observation: x})
        return actions

    def predict_boltzmann(self, x):
        q_values = self.sess.run(self.output, feed_dict={self.ph_observation: x})
        q_values = q_values.astype('float64')
        exp_values = np.exp(np.clip(q_values, -500, 500))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(self.num_actions), p=probs[0])
        return action

    def save(self, episode):
        pass

    def load(self):
        pass

    def _create_graph(self):
        self.ph_observation = tf.placeholder(tf.float32, shape=[None, self.observation_dim], name="Observation")
        self.ph_target = tf.placeholder(tf.float32, shape=[None], name="TDTarget")
        self.ph_action = tf.placeholder(tf.int32, shape=[None], name="Action")
        self.global_step = tf.Variable(0, trainable=False, name='step')

        self.inputs = tf.keras.Input(tensor=self.ph_observation)
        self.l1 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu, name="Layer1")(self.inputs)
        self.l2 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu, name="Layer2")(self.l1)
        self.l3 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu, name="Layer3")(self.l2)
        self.output = tf.keras.layers.Dense(units=self.num_actions, activation=None, name="Q_Out")(self.l3)
        self.model = tf.keras.Model(self.inputs, self.output)

        self.inputs_t = tf.keras.Input(tensor=self.ph_observation)
        self.l1_t = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)(self.inputs_t)
        self.l2_t = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)(self.l1_t)
        self.l3_t = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)(self.l2_t)
        self.output_t = tf.keras.layers.Dense(units=self.num_actions, activation=None)(self.l3_t)
        self.model_target = tf.keras.Model(self.inputs_t, self.output_t)

        self.max_output = tf.reduce_max(self.output_t, axis=1, name="MaxQValue")
        self.action = tf.argmax(self.output, axis=1, name="PredictAction")

        with tf.name_scope("Loss"):
            self.N = tf.shape(self.ph_action)[0]
            self.index = tf.stack([tf.range(self.N, dtype=tf.int32), self.ph_action], axis=1)
            self.yy = tf.gather_nd(self.output, self.index)
            self.loss = tf.losses.mean_squared_error(self.ph_target, self.yy)

        self.opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

    def _copy_to_target_nn(self):
        with self.graph.as_default():
            self.model_target.set_weights(.99 * np.array(self.model_target.get_weights()) +
                                          .01 * np.array(self.model.get_weights()))

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Loss", self.loss))
        # for var in tf.trainable_variables():
        #     summaries.append(tf.summary.histogram("weights_%s" % var.name, var))
        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter(self.log_dir, self.graph)


def q_learning(env_name, iter, render, gamma=.98, seed=0):
    env = gym.make(env_name)

    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = QNetwork(env_name, env.action_space.n, env.observation_space.shape[0], gamma, seed, log_dir='/tmp/qlearn/lr')

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
            action = model.predict_boltzmann([observation])

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
            model.train_batch(np.array(m_states)[indexs],
                              np.array(m_actions)[indexs],
                              np.array(m_rewards)[indexs],
                              np.array(m_next_states)[indexs],
                              np.array(m_done)[indexs])
            observation = new_observation

            if done:
                break
        print("epoch %d reward %d" % (i, epoch_reward))

    print("Average Reward: %d" % (total_reward / iter))
    return total_reward / iter


if __name__ == '__main__':
    epoch = 300
    gamma = 1.
    render = False
    env_name = 'CartPole-v0'

    seed = 10
    q_learning(env_name=env_name, iter=epoch, render=render, gamma=gamma, seed=seed)
