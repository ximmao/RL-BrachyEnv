from agents_tabular import WorldParameterTransformer, QLearning, SARSA
from MCworlds_tabular import DataGetter, MCenv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    num_runs = 500
    num_episode = 100

    env = WorldParameterTransformer(MCenv(DataGetter(), 5, 3, 100, False))
    all_rewards_Q = []
    all_lengths_Q = []
    final_greedy_rewards_Q = []
    final_greedy_lengths_Q = []
    for r in range(num_runs):
        agent = QLearning(env)
        #agent = SARSA(env)
        episode_returns = []
        episode_lengths = []
        for i in range(num_episode):
            cum_r, total_t = agent.RunEpisode()
            episode_returns.append(cum_r)
            episode_lengths.append(total_t)
        greedy_perf_rewards_Q, greedu_perf_lengths_Q = agent.RunGreedyPolicy()
        #print(episode_returns)
        all_rewards_Q.append(episode_returns)
        all_lengths_Q.append(episode_lengths)
        final_greedy_rewards_Q.append(greedy_perf_rewards_Q)
        final_greedy_lengths_Q.append(greedu_perf_lengths_Q)
    all_rewards_Q = np.array(all_rewards_Q)
    Q_learning_rewards = np.mean(all_rewards_Q, axis=0)
    all_lengths_Q = np.array(all_lengths_Q)
    Q_learning_lengths = np.mean(all_lengths_Q, axis=0)

    all_rewards_SARSA = []
    all_lengths_SARSA = []
    final_greedy_rewards_SARSA = []
    final_greedy_lengths_SARSA = []
    for r in range(num_runs):
        #agent = QLearning(env)
        agent = SARSA(env)
        episode_returns = []
        episode_lengths = []
        for i in range(num_episode):
            cum_r, total_t = agent.RunEpisode()
            episode_returns.append(cum_r)
            episode_lengths.append(total_t)
        greedy_perf_rewards_SARSA, greedu_perf_lengths_SARSA = agent.RunGreedyPolicy()
        #print(episode_returns)
        all_rewards_SARSA.append(episode_returns)
        all_lengths_SARSA.append(episode_lengths)
        final_greedy_rewards_SARSA.append(greedy_perf_rewards_SARSA)
        final_greedy_lengths_SARSA.append(greedu_perf_lengths_SARSA)
    all_rewards_SARSA = np.array(all_rewards_SARSA)
    SARSA_rewards = np.mean(all_rewards_SARSA, axis=0)
    all_lengths_SARSA = np.array(all_lengths_SARSA)
    SARSA_lengths = np.mean(all_lengths_SARSA, axis=0)


    print("\nfinal greedy performance:")
    print("Q-learning: reward", np.mean(final_greedy_rewards_Q), "length", np.mean(final_greedy_lengths_Q))
    print("SARSA: reward", np.mean(final_greedy_rewards_SARSA), "length", np.mean(final_greedy_lengths_SARSA))

    plt.plot([i+1 for i in range(num_episode)], Q_learning_rewards)
    plt.plot([i+1 for i in range(num_episode)], SARSA_rewards)
    plt.title("Performance in Rewards of Q-learning and SARSA in simplified discrete MC environment")
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Rewards')
    plt.legend(['Q-learning', 'SARSA'])
    plt.show()

    plt.plot([i+1 for i in range(num_episode)], Q_learning_lengths)
    plt.plot([i+1 for i in range(num_episode)], SARSA_lengths)
    plt.title("Performance in Lengths of Q-learning and SARSA in simplified discrete MC environment")
    plt.xlabel('Episodes')
    plt.ylabel('Average Episode Lengths')
    plt.legend(['Q-learning', 'SARSA'])
    plt.show()
