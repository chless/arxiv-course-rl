import random

import numpy as np
# import sklearn.gaussian_process as gp
# from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from approx.CustomGaussianProcessRegressor import CustomGaussianProcessRegressor as CustomGPR


class GPAgent:
    def __init__(self, num_actions, epsilon=0.2, gamma=0.8, magnitude=1.0, length_scale=1.0, alpha=1e-2, num_opt_iter=0, update_rate=0.1, random_seed=42):
        self.nA = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.num_opt_iter = num_opt_iter
        if self.num_opt_iter == 0:
            self.kernel = C(magnitude, constant_value_bounds="fixed") * \
                          RBF(length_scale, length_scale_bounds="fixed")
        else:
            self.kernel = C(magnitude) * RBF(length_scale)  # kernel parameter auto-optimized
        self.alpha = alpha
        self.update_rate = update_rate
        self.random_seed = random_seed
        self.value_fn_list = self._set_value_fn_list()
        self.history = [None for i in range(self.nA)]

    def _set_value_fn_list(self):
        value_fn_list = []
        for a in range(self.nA):
            # value_fn = GPR(kernel=self.kernel, alpha=self.alpha)
            value_fn = CustomGPR(kernel=self.kernel,
                                 alpha=self.alpha,
                                 normalize_y=True,
                                 n_restarts_optimizer=self.num_opt_iter,
                                 random_state=self.random_seed)
            value_fn_list += [value_fn]
        return value_fn_list

    def get_action(self, state, return_value=False):
        value_list = []
        for value_fn in self.value_fn_list:
            mean = value_fn.predict(state.reshape(1,-1), return_std=False).flatten()
            value_list += [mean]
        if random.uniform(0,1) < self.epsilon:
            best_action = np.random.choice(self.nA)
        else:
            if (max(value_list) - min(value_list)) < max(value_list) * 0.01:
                best_action = np.random.choice(self.nA)
            else:
                best_action = np.array(np.concatenate(value_list)).argmax()
        if return_value:
            return best_action, [value for value in value_list]
        else:
            return best_action

    def update_epsilon(self, epoch, num_vanish=100):
        num_vanish = min(num_vanish, 100)
        self.epsilon = self.epsilon_init * (1 - min(epoch+1, num_vanish) / num_vanish)

    def update(self, observation, num_step):
        state, action, reward, next_state, next_action, done = observation
        history = self.history[action]
        old_value = self.value_fn_list[action].predict(state.reshape(1, -1))
        one_step_value = self.value_fn_list[next_action].predict(next_state.reshape(1, -1))
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * one_step_value
        td_error = td_target - old_value
        new_value = old_value + self.update_rate / (num_step + 1) * td_error
        if history is None:
            history = {"state": state.reshape(1,-1), "value": new_value}
        else:
            history["state"] = np.append(history["state"], state.reshape(1,-1), 0)[-1000:]
            history["value"] = np.append(history["value"], new_value)[-1000:]

        self.update_value_fn(action, history["state"], history["value"])
        self.history[action] = history

    def rollout_update(self, observation):
        for state, action, reward in observation:
            history = self.history[action]
            if history is None:
                history = {"state": state.reshape(1, -1), "value": np.array([reward])}
            else:
                history["state"] = np.append(history["state"], state.reshape(1, -1), 0)
                history["value"] = np.append(history["value"], np.array([reward]))
            self.history[action] = history
        for action in range(self.nA):
            self.update_value_fn(action, self.history[action]["state"], self.history[action]["value"])


    def update_value_fn(self, action, x, y):
        # value_fn = GPR(kernel=self.kernel, alpha=self.alpha)
        # value_fn = CustomGPR(kernel=self.kernel, alpha=self.alpha, normalize_y=True)
        kernel = self.value_fn_list[action].kernel
        value_fn = CustomGPR(kernel=kernel, alpha=self.alpha, normalize_y=True, n_restarts_optimizer=0, random_state=self.random_seed)
        if len(y.shape) < 2:
            y = y.reshape(-1,1)
        value_fn.fit(x, y)
        self.value_fn_list[action] = value_fn
        # self.value_fn_list[action].fit(x, y)