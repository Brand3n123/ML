# Comparison: UCB vs Column-Sum Greedy
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
N = min(10000, dataset.shape[0])
d = dataset.shape[1]

# --- UCB ---
ads_selected_ucb = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward_ucb = 0
cumulative_rewards_ucb = []
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected_ucb.append(ad)
    numbers_of_selections[ad] += 1
    reward = int(dataset.values[n, ad])
    sums_of_rewards[ad] += reward
    total_reward_ucb += reward
    cumulative_rewards_ucb.append(total_reward_ucb)

# --- Column-sum greedy (offline) ---
column_sums = dataset.sum(axis=0).values
best_ad = int(np.argmax(column_sums))
# If we always pick best_ad for each round
ads_selected_greedy = [best_ad] * N
rewards_greedy = dataset.iloc[:N, best_ad].astype(int).values
cumulative_rewards_greedy = np.cumsum(rewards_greedy)
total_reward_greedy = int(cumulative_rewards_greedy[-1])

# Save a histogram comparing selections
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(ads_selected_ucb, bins=np.arange(d+1)-0.5, rwidth=0.8)
plt.title('UCB: ads selected')
plt.xlabel('Ad')
plt.ylabel('Selections')

plt.subplot(1,2,2)
plt.bar(range(d), [N if i==best_ad else 0 for i in range(d)])
plt.title('Greedy: always select best ad')
plt.xlabel('Ad')
plt.ylabel('Selections')
plt.tight_layout()
plt.savefig('ucb_vs_greedy_hist.png')

# Save cumulative reward plot
plt.figure()
plt.plot(range(1, N+1), cumulative_rewards_ucb, label='UCB')
plt.plot(range(1, N+1), cumulative_rewards_greedy, label='Greedy (column-sum)')
plt.xlabel('Round')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.grid(True)
plt.savefig('ucb_vs_greedy_cumulative.png')

# Print summary
print(f'UCB total reward: {total_reward_ucb}')
print(f'Greedy (column-sum) best ad: {best_ad}')
print(f'Greedy total reward: {total_reward_greedy}')
print(f'Difference (Greedy - UCB): {total_reward_greedy - total_reward_ucb}')

# Also write a tiny numeric comparison to a text file
with open('ucb_vs_greedy_summary.txt', 'w') as f:
    f.write(f'UCB total reward: {total_reward_ucb}\n')
    f.write(f'Greedy best ad: {best_ad}\n')
    f.write(f'Greedy total reward: {total_reward_greedy}\n')
    f.write(f'Difference (Greedy - UCB): {total_reward_greedy - total_reward_ucb}\n')

print('Saved plots: ucb_vs_greedy_hist.png, ucb_vs_greedy_cumulative.png')
print('Saved summary: ucb_vs_greedy_summary.txt')
