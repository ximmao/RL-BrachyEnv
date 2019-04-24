import numpy as np
from MCworlds_tabular import MCenv, DataGetter

class WorldParameterTransformer():
    def __init__(self, mc_world):
        self.world = mc_world
        self.n_a = self.world.action_space.n
        self.n_f = self.world.feature_space.n
        if self.world.include_env:
            pass
        else:
            self.n_s = self.world.feature_space.feature_range ** self.n_f
            self.n_w = self.n_s * self.n_a
        #print(self.n_s, self.n_a, self.n_f, self.n_w)

    def reset(self):
        # get random state & action
        mc_feature = self.world.reset()
        self.s = self.convert_feature(mc_feature)
        #a = np.zeros(self.n_a)
        #a[int(np.random.randint(self.n_a))] = 1.

        return self.s, np.random.randint(self.n_a)

    def step(self, action):
        # action is one_hot action vector
        #mc_action = np.argmax(action)
        mc_feature, reward, done, success = self.world.step(action)
        self.s = self.convert_feature(mc_feature)
        return self.s, reward, done, success

    def convert_feature(self, mc_feature):
        #print(mc_feature)
        index = 0
        multiplier = 1
        for v in mc_feature:
            if v == 100.0:
                index += 0 * multiplier
            elif v == 200.0:
                index += 1 * multiplier
            elif v == 300.0:
                index += 2 * multiplier
            multiplier *= 3
        return index

class BaseAgent():
    def __init__(self, world, gamma = 0.9, alpha = 0.9, epsilon = 0.1):
        self.world = world
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def InitAll(self):
        # initialize w
        self.W = np.zeros((self.world.n_s, self.world.n_a)).reshape(-1)

    def GetCurrentPolicy(self, q_sa):
        # Get epsilon-greedy policy based on Q value
        # policy is n*n * n_a matrix
        #self.Pi = np.zeros((len(self.grid_world.A), self.grid_world.R.shape[0])) + 1.0 / len(self.grid_world.A)
        q_sa_max = np.max(q_sa)
        policy = (self.epsilon / self.world.n_a) * np.ones(self.world.n_a).astype(np.float32)
        q_sa_max_bool = (q_sa == q_sa_max).astype(np.float32)
        policy += q_sa_max_bool * ((1 - self.epsilon) / np.sum(q_sa_max_bool))
        assert np.sum(policy) > (1 - 0.000001) and np.sum(policy) < (1 + 0.000001)
        return policy

    def GetGreedyPolicy(self, q_sa):
        # Get greedy policy based on Q value
        # policy is n*n * n_a matrix
        #self.Pi = np.zeros((len(self.grid_world.A), self.grid_world.R.shape[0])) + 1.0 / len(self.grid_world.A)
        q_sa_max = np.max(q_sa)
        #policy = (self.epsilon / self.world.n_a) * np.ones(self.world.n_a).astype(np.float32)
        q_sa_max_bool = (q_sa == q_sa_max).astype(np.float32)
        policy = q_sa_max_bool * (1 / np.sum(q_sa_max_bool))
        assert np.sum(policy) > (1 - 0.000001) and np.sum(policy) < (1 + 0.000001)
        return policy

    def RunEpisode(self):
        raise NotImplementedError

    def SampleAction(self, state):
        # assuming policy is a num_state * num_action matrix
        #print("sample state", state, "policy", policy[state, :])
        #q_sa = np.zeros(self.world.n_a)
        #for i in range(q_sa.shape[0]):
        #    a = np.zeros(self.n_a)
        #    a[i] = 1
        #    q_sa[i] = self.GetQValue(state, a)
        q_sa = self.GetQValue(state, "all")
        policy = self.GetCurrentPolicy(q_sa)
        optimum_idx = np.argmax(policy)
        r = np.random.rand()
        #print(r)
        acc = 0.0
        for i in range(policy.shape[0]):
            if not i == optimum_idx:
                #print(acc, acc + policy[state, i], "action", i)
                if r >= acc and r < (acc + policy[i]):
                    return i
                acc += policy[i]
        return optimum_idx

    def SampleGreedyAction(self, state):
        # assuming policy is a num_state * num_action matrix
        #print("sample state", state, "policy", policy[state, :])
        #q_sa = np.zeros(self.world.n_a)
        #for i in range(q_sa.shape[0]):
        #    a = np.zeros(self.n_a)
        #    a[i] = 1
        #    q_sa[i] = self.GetQValue(state, a)
        q_sa = self.GetQValue(state, "all")
        policy = self.GetGreedyPolicy(q_sa)
        optimum_idx = np.argmax(policy)
        r = np.random.rand()
        #print(r)
        acc = 0.0
        for i in range(policy.shape[0]):
            if not i == optimum_idx:
                #print(acc, acc + policy[state, i], "action", i)
                if r >= acc and r < (acc + policy[i]):
                    return i
                acc += policy[i]
        return optimum_idx

    def RunGreedyPolicy(self):
        c_s, c_a = self.world.reset()
        #print("curr_state", curr_state, "policy", policy[curr_state, :])
        cumu_returns = 0.0
        t = 0
        while True:
            n_s, r, done, success = self.world.step(c_a)
            cumu_returns += r
            if done:
                #print(self.W)
                n_a = self.SampleGreedyAction(n_s)
                #print("Episode finished after {} timesteps".format(t+1))
                break
            #print("next_state", observation, "reward", reward)
            n_a = self.SampleGreedyAction(n_s)
            c_s = n_s
            c_a = n_a
            t+=1
        return cumu_returns, t+1

    def GetQValue(self, state, a = "all"):
        #print(state, a)
        # assume state and action both indice
        if isinstance(a,(str,)) and a == "all":
            q_sa = np.zeros(self.world.n_a)
            for i in range(q_sa.shape[0]):
                q_sa[i] = self.GetQValue(state, i)
            #print(q_sa)
            return q_sa
        else:
            feature_vec = np.zeros((self.world.n_s, self.world.n_a))
            feature_vec[state, a] = 1
            #print(feature_vec, self.W)
            return float(np.dot(self.W, feature_vec.reshape(-1)))

class SARSA(BaseAgent):
    def __init__(self, world, gamma = 1.0, alpha = 0.6, epsilon = 0.001, random_init = False):
        super().__init__(world, gamma, alpha, epsilon)
        self.InitAll()

    def UpdateParam(self, c_s, c_a, n_s, n_a, r, is_next_t):
        # update w
        temp = np.zeros((self.world.n_s, self.world.n_a))
        temp[c_s, c_a] = 1
        if is_next_t:
            G = r - self.GetQValue(c_s, c_a)
        else:
            G = r + self.gamma * self.GetQValue(n_s, n_a) - self.GetQValue(c_s, c_a)
        self.W = self.W + self.alpha * G * temp.reshape(-1)

    def RunEpisode(self):
        # episode is a list of trajectories
        # Q(s,a) with TD(0)
        c_s, c_a = self.world.reset()
        #print("curr_state", curr_state, "policy", policy[curr_state, :])
        cumu_returns = 0.0
        t = 0
        while True:
            n_s, r, done, success = self.world.step(c_a)
            cumu_returns += r
            if done:
                #print(self.W)
                n_a = self.SampleAction(n_s)
                self.UpdateParam(c_s, c_a, n_s, n_a, r, success)
                #print("Episode finished after {} timesteps".format(t+1))
                break
            #print("next_state", observation, "reward", reward)
            n_a = self.SampleAction(n_s)
            self.UpdateParam(c_s, c_a, n_s, n_a, r, success)
            c_s = n_s
            c_a = n_a
            t+=1
        return cumu_returns, t+1

class QLearning(BaseAgent):
    def __init__(self, world, gamma = 1.0, alpha = 0.9, epsilon = 0.1, random_init = False):
        super().__init__(world, gamma, alpha, epsilon)
        self.InitAll()

    def UpdateParam(self, c_s, c_a, n_s, n_a, r, is_next_t):
        # update w
        temp = np.zeros((self.world.n_s, self.world.n_a))
        temp[c_s, c_a] = 1
        if is_next_t:
            #print(r, "c_sa", self.GetQValue(c_s, c_a))
            G = r - self.GetQValue(c_s, c_a)
        else:
            #print(r, "n_s", self.GetQValue(n_s, "all"), "c_sa", self.GetQValue(c_s, c_a))
            G = r + self.gamma * np.max(self.GetQValue(n_s, "all")) - self.GetQValue(c_s, c_a)
        #print("G", G, "W", self.W, "f", temp.reshape(-1))
        self.W = self.W + self.alpha * G * temp.reshape(-1)
        #print(self.W)

    def RunEpisode(self):
        # episode is a list of trajectories
        # Q(s,a) with TD(0)
        c_s, c_a = self.world.reset()
        #print("curr_state", curr_state, "policy", policy[curr_state, :])
        #print("c_s", c_s, "c_a", c_a)
        cumu_returns = 0.0
        t = 0
        while True:
            n_s, r, done, success = self.world.step(c_a)
            cumu_returns += r
            if done:
                #print(self.W)
                self.UpdateParam(c_s, c_a, n_s, None, r, success)
                #print("Episode finished after {} timesteps".format(t+1))
                break
            #print("next_state", observation, "reward", reward)
            self.UpdateParam(c_s, c_a, n_s, None, r, success)
            n_a = self.SampleAction(n_s)
            c_s = n_s
            c_a = n_a
            #print("c_s", c_s, "c_a", c_a)
            t+=1
        return cumu_returns, t+1
