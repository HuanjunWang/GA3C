# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from multiprocessing import Queue

import time

from Config import Config
from ProcessAgent import ProcessAgent
from ThreadDynamicAdjustment import ThreadDynamicAdjustment
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer
from QNetwork import QNetwork
from Environment import Environment


class Server:
    def __init__(self):
        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.env = Environment()
        self.action_dim = self.env.get_num_actions()
        self.observation_dim = self.env.get_observation_dim()

        self.model = QNetwork(model_name="Name",
                              num_actions=self.action_dim,
                              observation_dim=self.observation_dim,
                              gamma=Config.DISCOUNT,
                              seed=Config.SEED,
                              log_dir=Config.LOG_DIR)

        self.training_step = 0
        self.action_step = 0

        self.agents = []
        self.predictors = []
        self.trainers = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)

    def add_agent(self):
        self.agents.append(
            ProcessAgent(len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q))
        self.agents[-1].start()

    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].join()
        self.agents.pop()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join()
        self.trainers.pop()

    def train_model(self, state, action, next_state, reward, done, trainer_id):
        self.model.train_batch(state=state,
                               action=action,
                               reward=reward,
                               next_state=next_state,
                               done=done)
        self.training_step += 1
        self.action_step += state.shape[0]

        self.dynamic_adjustment.temporal_training_count += 1

    def save_model(self):
        self.model.save(self.action_step)

    def main(self):
        self.dynamic_adjustment.start()

        learning_rate_multiplier = (Config.LEARNING_RATE_END - Config.LEARNING_RATE_START) / Config.ACTION_STEPS

        while self.action_step < Config.ACTION_STEPS:
            self.model.learning_rate = Config.LEARNING_RATE_START + learning_rate_multiplier * self.action_step
            time.sleep(0.01)

        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()


if __name__ == "__main__":
    server = Server()
    server.main()
