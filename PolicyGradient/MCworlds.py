import numpy as np
import torch
from pg_data.Utils import generate_contour_fixed, generate_plan_dict, generate_dose_dict, matrix_size, D

class FeatureTransformer():
    def __init__(self, prefix="./pg_data"):
        self.prefix = prefix
        self.contour = generate_contour_fixed(self.prefix)
        self.plan_dict = generate_plan_dict(self.prefix)

    def generate_plan_with_dwelltime(self, time_per_point):
        """
        Given dwell duration list, generate corresponding plan matrix
        """
        assert isinstance(time_per_point, list)
        assert len(time_per_point) == 10
        total_time = np.sum(time_per_point)
        total_plan = np.zeros(matrix_size)
        for idx in range(1, 11):
            total_plan += (self.plan_dict["plan_"+str(idx)]*time_per_point[idx-1])
        assert np.count_nonzero(total_plan) == 10, np.count_nonzero(total_plan)
        assert np.sum(total_plan) == total_time, np.sum(total_plan)
        return total_plan

    def get_nn_input(self, feature):
        # input torch Tensor, return concatenated matrix
        # compatible with batch input
        supposed_size = (2, *matrix_size)
        if len(feature.size()) == 2:
            is_first = True
            for f in feature:
                plan = self.generate_plan_with_dwelltime(f.cpu().numpy().reshape(-1).tolist())
                cnn_input = np.concatenate((self.contour.reshape(-1, *self.contour.shape), plan.reshape(-1, *plan.shape)), axis=0)
                if is_first:
                    batch_input = cnn_input.reshape(-1, *(cnn_input.shape))
                    is_first = False
                else:
                    batch_input = np.concatenate((batch_input, cnn_input.reshape(-1, *(cnn_input.shape))), axis=0)
            assert batch_input.shape == (feature.size(0), *supposed_size), batch_input.shape
            return torch.from_numpy(batch_input).float()
        elif len(feature.size()) == 1:
            plan = self.generate_plan_with_dwelltime(feature.cpu().numpy().reshape(-1).tolist())
            cnn_input = np.concatenate((self.contour.reshape(-1, *self.contour.shape), plan.reshape(-1, *plan.shape)), axis=0)
            cnn_input = cnn_input.reshape(-1, *(cnn_input.shape))
            assert cnn_input.shape == (1, *supposed_size), cnn_input.shape
            return torch.from_numpy(cnn_input).float()

class MCDataGetter():
    def __init__(self, directory="./pg_data"):
        self.prefix = "./pg_data"
        self.contour = generate_contour_fixed(self.prefix)
        self.dose_dict = generate_dose_dict(self.prefix)
        self.contour_index = [2, 3, 4] # generate_contour_fixed generate ct contour with 2 as CTV, 3 as OAR1 and 4 as OAR2
        assert isinstance(self.contour_index, list)
        assert len(self.contour_index) == 3

    def compute_dose_with_feature(self, feature):
        dose_map = np.zeros(matrix_size)
        for idx in range(1, 11):
            # each dose file is the dose when dwell time is 100
            dose_map += self.dose_dict["dose_"+str(idx)] * (float(feature[idx-1])/100.)
        assert dose_map.shape == matrix_size, dose_map.shape

        dose_dict = {}
        dose_dict['CTV'] = (self.contour == float(self.contour_index[0]))
        dose_dict['OAR-1'] = (self.contour == float(self.contour_index[1]))
        dose_dict['OAR-2'] = (self.contour == float(self.contour_index[2]))

        max_v = np.max(dose_map)
        ctv_dose = dose_map[dose_dict['CTV']]
        oar1_dose = dose_map[dose_dict['OAR-1']]
        oar2_dose = dose_map[dose_dict['OAR-2']]

        return D(ctv_dose, max_v, 90), D(oar1_dose, max_v, 50), D(oar2_dose, max_v, 50)

    def get_statistic(self, feature):
        CTV, OAR1, OAR2 = self.compute_dose_with_feature(feature)
        return CTV, OAR1, OAR2

class ActionSpace():
    def __init__(self, num_dwell, num_stats = 0, include_env = False):
        # continous action space

        if include_env:
            self.n = num_dwell + num_stats
        else:
            self.n = num_dwell
        self.v_max = 60.
        self.v_min = -60.

    def get_action_vec(self, action):
        if isinstance(action, list):
            assert len(action) == self.n
            return np.array(action)
        elif isinstance(action, np.ndarray):
            assert action.shape[0] == self.n
            return action

    def sample(self):
        # sample a action vector with value ranging from v_min to v_max
        vec = np.random.rand(self.n) * (self.v_max - self.v_min)
        vec += self.v_min
        return vec

class StateSpace():
    def __init__(self, number, number_s):
        self.n = number
        #assert self.n == len(state_vector_mapping)
        self.mapping_dict = None

    def get_state_vec(self, state):
        return self.mapping_dict[str(state)]

    def sample(self):
        return np.random.randint(self.n)

class FeatureSpace():
    # feature vec represent dwell time for each locations

    def __init__(self, num_dwell):
        self.v_min = 1.
        self.v_max = 200.
        self.n = num_dwell

    def sample(self):
        vec = np.random.rand(self.n) * (self.v_max - self.v_min)
        vec += self.v_min
        return vec

    def get_feature_vec(self,feature):
        if isinstance(feature, list):
            assert len(feature) == self.n
            return np.array(feature)
        elif isinstance(feature, np.ndarray):
            assert feature.shape[0] == self.n
            return feature

    def fit(self, vec):
        for i in range(vec.shape[0]):
            if vec[i] < self.v_min:
                vec[i] = self.v_min
            if vec[i] > self.v_max:
                vec[i] = self.v_max
        return vec

class MCenv():
    def __init__(self, data_getter, n_d, n_s = 3, H = 10, constraint_idx = 0, include_env = False):
        self.feature_space = FeatureSpace(n_d)
        self.action_space = ActionSpace(n_d, n_s, include_env)
        self.include_env = include_env
        if self.include_env:
            self.state_space = StateSpace(n_d, n_s)
            self.observation_space = self.state_space
        else:
            self.observation_space = self.feature_space

        self.s = None
        self.counter = 0
        self.horizon = H
        self.constraint_idx = constraint_idx
        self.data_getter = data_getter

    def step(self, action):
        curr_feature = self.s[: self.feature_space.n]
        next_feature = self.feature_space.fit(self.feature_space.get_feature_vec(curr_feature) + self.action_space.get_action_vec(action))
        statistic_raw = self.data_getter.get_statistic(next_feature)
        self.s = self.build_state(next_feature, statistic_raw, self.include_env)
        reward = self.get_reward(statistic_raw, self.constraint_idx)
        self.counter += 1
        done = False
        success = False
        if self.check_stats(statistic_raw, self.constraint_idx) or self.counter >= self.horizon:
            done = True
        if self.check_stats(statistic_raw, self.constraint_idx) and self.counter < self.horizon:
            success = True
        if success == True:
            assert done == True

        return self.s, reward, done, success

    def reset(self):
        # sample initial state
        while True:
            next_feature = self.feature_space.sample()
            statistic_raw = self.data_getter.get_statistic(next_feature)
            if not self.check_stats(statistic_raw, self.constraint_idx):
                # not terminal state
                self.prev_stats = statistic_raw
                break
        self.s = self.build_state(next_feature, statistic_raw, self.include_env)
        self.counter = 0
        return self.s

    def build_state(self, feature, statistic, include_env = False):
        if include_env:
            return np.concatenate((feature, statistic))
        else:
            return feature

    def check_stats(self, statistic, constraint_idx):
        if constraint_idx == 0:
            if float(statistic[0]) >= 4.0 and float(statistic[0]) < 8.0 and float(statistic[1]) < 1.2 and float(statistic[2]) < 1.2:
                return True
            else:
                return False
        elif constraint_idx == 1:
            if float(statistic[0]) >= 5.0 and float(statistic[0]) < 7.0 and float(statistic[1]) < 1.0 and float(statistic[2]) < 1.0:
                return True
            else:
                return False
        elif constraint_idx == 2:
            if float(statistic[0]) >= 3.0 and float(statistic[0]) < 5.0 and float(statistic[1]) >= 0.8 and float(statistic[2]) >= 0.8:
                return True
            else:
                return False


    def get_reward(self, statistic, constraint_idx):
        #if statistic[0] > 15. and statistic[1] < 5. and statistic[2] < 5:
        if constraint_idx == 0:
            if float(statistic[0]) >= 4.0 and float(statistic[0]) < 8.0 and float(statistic[1]) < 1.2 and float(statistic[2]) < 1.2:
                return 20
            else:
                return -1
        elif constraint_idx == 1:
            if float(statistic[0]) >= 5.0 and float(statistic[0]) < 7.0 and float(statistic[1]) < 1.0 and float(statistic[2]) < 1.0:
                return 20
            else:
                return -1
        elif constraint_idx == 2:
            if float(statistic[0]) >= 3.0 and float(statistic[0]) < 5.0 and float(statistic[1]) >= 0.8 and float(statistic[2]) >= 0.8:
                return 20
            else:
                return -1

if __name__ == "__main__":
    if True:
        env = MCenv(MCDataGetter(), 10, 3, 50, False)
        print("initial state", env.reset())
        while True:
            a = env.action_space.sample()
            print("action", env.action_space.get_action_vec(a))
            n_s, r, done, success = env.step(a)
            print(n_s, r)
            if done:
                break
    if False:
        getter = MCDataGetter()
        print(getter.get_statistic([100,200,300,200,100]))
        env = MCenv(MCDataGetter(), 5, 3, 10, False)
        for x1 in [100.0, 200.0, 300.0]:
            for x2 in [100.0, 200.0, 300.0]:
                for x3 in [100.0, 200.0, 300.0]:
                    for x4 in [100.0, 200.0, 300.0]:
                        for x5 in [100.0, 200.0, 300.0]:
                            #feature = [int(x1), int(x2), int(x3), int(x4), int(x5)]
                            feature = [x1, x2, x3, x4, x5]
                            if env.get_reward(getter.get_statistic(feature)) != -1:
                                print(list(map(int,feature)), getter.get_statistic(feature))
