import numpy as np

class DataGetter():
    def __init__(self, directory = "statistics.txt"):
        with open(directory, "r") as my_in:
            self.statistic_mapping = {}
            for line in my_in:
                pair = line.strip().split(":")
                self.statistic_mapping[pair[0]] = list(map(float, pair[1].split(",")))
    def get_statistic(self, feature):
        # convert to int as statistics key is written in string-ed int
        int_feature = list(map(int, feature))
        name = ",".join(list(map(str, int_feature)))
        return self.statistic_mapping[name]

class ActionSpace():
    def __init__(self, num_dwell, num_stats, include_env):
        # 2 actions per dwell location

        per_loc_action_list = [-1., 1.]
        self.n = num_dwell * len(per_loc_action_list)
        if include_env:
            dim = num_dwell + num_stats
        else:
            dim = num_dwell
        self.mapping_dict = {}
        for i in range(num_dwell):
            temp = np.zeros(dim)
            temp[i] = 100.
            for j in range(len(per_loc_action_list)):
                self.mapping_dict[str(i * len(per_loc_action_list) + j)] = temp * per_loc_action_list[j]
        assert self.n == len(self.mapping_dict)

    def get_action_vec(self, action):
        #print(self.mapping_dict[str(action)])
        return self.mapping_dict[str(action)]

    def sample(self):
        return np.random.randint(self.n)

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
    def __init__(self, num_dwell):
        self.v_min = 100
        self.v_max = 300
        self.interval = 100
        self.feature_range = int((self.v_max - self.v_min) / self.interval + 1)
        #self.n = int(np.power(((self.v_max - self.v_min) / self.interval + 1), num_dwell))
        #temp = np.array([i+1. for i in range(v_min, v_max + 1)])
        #self.mapping_dict = {}
        #if num_dwell == 2:
        #    for i in range(temp.shape[0]):
        #        for j in range(temp.shape[0]):
        #            self.mapping_dict[str(i + len(temp)*j)] = np.array([temp[i], temp[j]])
        #assert self.n = len(self.mapping_dict)
        self.n = num_dwell

    #def get_feature_vec(self, feature):
    #    return self.mapping_dict[str(feature)]

    def sample(self):
        return np.array([float(self.v_min + self.interval * np.random.randint(0, self.feature_range)) for i in range(self.n)])

    #def get_index(self, feature):
        #temp = 0
        #for i in range(feature.shape[0]):
        #    temp += int(((int(feature[i]) - 1) % self.feature_range) * np.power(self.feature_range, i))

        #assert self.mapping_dict[str(temp)].tolist() == feature.tolist()
        #return temp

    def fit(self, vec):
        for i in range(vec.shape[0]):
            if vec[i] < 100.0:
                vec[i] = 100.0
            if vec[i] > 300.0:
                vec[i] = 300.0
        return vec

class MCenv():
    def __init__(self, data_getter, n_d, n_s, H, include_env):
        self.feature_space = FeatureSpace(n_d)
        self.action_space = ActionSpace(n_d, n_s, include_env)
        self.include_env = include_env
        if self.include_env:
            self.state_space = StateSpace(n_d, n_s)
            self.observation_space = self.state_space
        else:
            self.observation_space = self.feature_space

        #self.list_terminal = []
        self.s = None
        self.counter = 0
        self.horizon = H
        self.data_getter = data_getter

    def step(self, action):
        curr_feature = self.s[: self.feature_space.n]
        #print(curr_feature + self.action_space.get_action_vec(action))
        next_feature = self.feature_space.fit(curr_feature + self.action_space.get_action_vec(action))
        statistic_raw = self.data_getter.get_statistic(next_feature)
        #statistic = self.state_space.fit(statistic_raw)
        self.s = self.build_state(next_feature, statistic_raw, self.include_env)
        reward = self.get_reward(self.prev_stats, statistic_raw)
        self.prev_stats = statistic_raw
        self.counter += 1
        done = False
        success = False
        if self.check_stats(statistic_raw) or self.counter >= self.horizon:
            done = True
        if self.check_stats(statistic_raw) and self.counter < self.horizon:
            success = True
        if success == True:
            assert done == True

        return self.s, reward, done, success

    def reset(self):
        # sample initial state
        while True:
            next_feature = self.feature_space.sample()
            #print(type(next_feature))
            statistic_raw = self.data_getter.get_statistic(next_feature)
            #statistic = self.state_space.fit(statistic_raw)
            if not self.check_stats(statistic_raw):
                # not terminal state
                self.prev_stats = statistic_raw
                break
        self.s = self.build_state(next_feature, statistic_raw, self.include_env)
        self.counter = 0
        #if self.include_env:
        #    self.s_idx = self.state_space(self.s)
        #else:
        #    self.s_idx = self.feature_space(self.s)
        #print(type(self.s))
        return self.s

    def build_state(self, feature, statistic, include_env):
        if include_env:
            return np.concatenate((feature, statistic))
        else:
            return feature

    def check_stats(self, statistic):
        if float(statistic[0]) >= 5.0 and float(statistic[0]) < 8.0 and float(statistic[1]) < 1.2 and float(statistic[2]) < 1.2:
            return True
        else:
            return False

    def get_reward(self, prev_stats, statistic):
        #if statistic[0] > 15. and statistic[1] < 5. and statistic[2] < 5:
        if float(statistic[0]) >= 5.0 and float(statistic[0]) < 8.0 and float(statistic[1]) < 1.2 and float(statistic[2]) < 1.2:
            return 20
        else:
            return -1

if __name__ == "__main__":
    if False:
        env = MCenv(DataGetter(), 5, 3, 10, False)
        print(env.reset())
        while True:
            a = np.random.randint(10)
            print(a, env.action_space.get_action_vec(a))
            n_s, r, done, success = env.step(a)
            print(n_s, r)
            if done:
                break
    if True:
        getter = DataGetter()
        print(getter.get_statistic([100,200,300,200,100]))
        env = MCenv(DataGetter(), 5, 3, 10, False)
        for x1 in [100.0, 200.0, 300.0]:
            for x2 in [100.0, 200.0, 300.0]:
                for x3 in [100.0, 200.0, 300.0]:
                    for x4 in [100.0, 200.0, 300.0]:
                        for x5 in [100.0, 200.0, 300.0]:
                            #feature = [int(x1), int(x2), int(x3), int(x4), int(x5)]
                            feature = [x1, x2, x3, x4, x5]
                            if env.get_reward(getter.get_statistic(feature)) != -1:
                                print(list(map(int,feature)), getter.get_statistic(feature))
