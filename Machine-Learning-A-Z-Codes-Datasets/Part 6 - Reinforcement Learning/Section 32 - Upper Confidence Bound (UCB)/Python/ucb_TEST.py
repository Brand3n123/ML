"""
-We repeatedly choose one ad per user by scoring each ad with UCB: its observed click rate so far plus
an uncertainty bonus that makes under-tested ads worth trying. After each choice, we record whether that ad got
clicked, update that ad's counts/rewards, and use the new information to make the next choice smarter.

-And the purpose of UCB is to most efficiently determine the highest confidence
in the minimum (or specifically assigned) number of rounds/iterations?

-Close, but I'd phrase it slightly differently:
The purpose of UCB is to efficiently find and exploit the best option while spending as few rounds as possible on weaker options.
It does not just seek "highest confidence"; it uses confidence/uncertainty to balance testing unknown ads
against choosing the ad that currently looks best.
In this script: within N = 1000 rounds, UCB tries to maximize total clicks by learning which ad performs best
as it goes.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

# Each row is one user/round. Each ad column contains the result we would get
# if that ad were shown to that user: 1 = clicked, 0 = not clicked.
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000  # number of rounds/users to simulate
d = 10    # number of ads/options

# Bookkeeping for what UCB has learned so far.
ads_selected = []              # one selected ad index per round; ends with length N
number_of_selections = [0] * d # per-ad selection counts; values should add up to N
sums_of_rewards = [0] * d      # per-ad click totals from selected rounds
total_reward = 0               # total clicks collected by the whole strategy

# UCB balances:
# - exploitation: choose ads with strong observed click rates
# - exploration: keep testing ads that are still uncertain/under-tested
for n in range(0, N):
    selected_ad = 0
    max_upper_bound = 0

    # Score each ad, then keep the ad with the highest score.
    for candidate_ad in range(0, d):
        if number_of_selections[candidate_ad] > 0:
            # Average reward is this ad's observed click rate so far.
            average_reward = sums_of_rewards[candidate_ad] / number_of_selections[candidate_ad]

            # delta_i is the exploration bonus. It is larger when this ad has
            # been selected fewer times, and shrinks as we gain more data.
            delta_i = math.sqrt((3 / 2) * (math.log(n + 1) / number_of_selections[candidate_ad]))

            # UCB score = what we know so far + uncertainty bonus.
            upper_bound = average_reward + delta_i
        else:
            # First-time ads cannot have an average yet. This huge value forces
            # every ad to be tried at least once before the formula takes over.
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            selected_ad = candidate_ad

    # Apply the selected ad to this round/user and update the bookkeeping.
    ads_selected.append(selected_ad)
    number_of_selections[selected_ad] += 1

    # reward is only this round's click result for the selected ad.
    reward = dataset.values[n, selected_ad]
    sums_of_rewards[selected_ad] += reward
    total_reward += reward

# Visualize how often each ad was selected.
# This opens the graph window and also saves a PNG copy in this folder.
plt.hist(ads_selected)
"""plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.savefig('ucb_histogram.png')"""
plt.show()
