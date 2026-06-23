import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 1000
d = 10
ads_selected = []
number_of_selcetions = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0,N):
    print('it worked')