# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 09:52:00 2023

@author: Daniel
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "lunar_lander_learning_rate_noise_test.csv"


def main():
    results_df = pd.read_csv(RESULTS)
    bo_df, rs_df = results_df.groupby(results_df.method)
    bo_df = bo_df[1]
    print(bo_df)
    bo_df = bo_df.sort_values(by="learning_rate")
    print(bo_df)
    exp = 0
    for bo_experiment in bo_df.groupby(bo_df.experiment):
        exp = bo_experiment[1]
        plt.plot(exp.learning_rate, exp.reward_lower_bound, label=f"Experiment {exp}")

    plt.xlabel("Learning rate")
    plt.ylabel("Reward lower bound")
    plt.title("Objective functions")
    plt.savefig("Objective function low fidelity lunar lander learning rate")
if __name__ == "__main__":
    main()