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

from threading import Thread
import numpy as np

from Config import Config


class ThreadTrainer(Thread):
    def __init__(self, server, id, observation_dim):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.observation_dim = observation_dim
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            batch_size = 0

            state__ = []
            action__ = []
            next_state__ = []
            reward__ = []
            done__ = []
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                batch_size += 1
                state_, action_, next_state_, reward_, done_ = self.server.training_q.get()
                state__.append(state_)
                action__.append(action_)
                next_state__.append(next_state_)
                reward__.append(reward_)
                done__.append(done_)
            
            if Config.TRAIN_MODELS:
                self.server.train_model(state=np.array(state__),
                                        action=np.array(action__),
                                        next_state=np.array(next_state__),
                                        reward=np.array(reward__),
                                        done=np.array(done__),
                                        id=self.id)
