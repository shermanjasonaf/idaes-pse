#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-12-01 Wed 16:29:50

@author: jasherma

Uncertainty models for multiperiod rankine model LMP's.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from pyomo.contrib.pyros import uncertainty_sets


def get_lmp_data(n_time_points, start=0):
    """
    Extract LMP signal data from file for first `n_time_points`
    time periods, beginning with period `start`.
    """
    assert type(n_time_points) is int
    assert type(start) is int

    # read lmp signal (all 8736 hrs) data from file
    with open('./rts_results_all_prices.npy', 'rb') as f:
        _ = np.load(f)  # dispatch levels
        prices = np.load(f)  # lmp signal data

    # max out prices at $200 / MWH
    prices_used = copy.copy(prices)
    prices_used[prices_used > 200] = 200

    # plt.bar(np.arange(prices_used.size), prices_used)
    # plt.show()

    # return appropriate portion of LMP data.
    return prices_used[start:n_time_points + start]


class SimpleLMPBoxSet():
    def __init__(self, lmp_data, n_recent, growth_rate, avg_multiplier):
        """Initialize simple LMP box set."""
        self.lmp_sig_nom = lmp_data
        self.n_time_points = len(lmp_data)
        self.n_recent = n_recent
        self.growth_rate = growth_rate
        self.mov_avg_multiplier = avg_multiplier

    def bounds(self):
        """Evaluate LMP uncertainty set bounds."""
        multipliers = [self.growth_rate * t for t in range(self.n_time_points)]

        moving_average = [self.mov_avg_multiplier
                          * sum([self.lmp_sig_nom[tprime]
                                / (t - max(0, t - self.n_recent))
                                for tprime in range(max(0, t - self.n_recent),
                                                    t - 1)])
                          for t in range(self.n_time_points)]
        lmp_bounds = [(max(0, self.lmp_sig_nom[time]
                           * (1 - multipliers[time])),
                      max(moving_average[time], self.lmp_sig_nom[time] *
                          (1 + multipliers[time])))
                      for time in range(self.n_time_points)]

        return lmp_bounds

    def write_bounds(self, filename):
        """Write box set bounds to file."""
        bounds = self.bounds()

        # resolve bounds into lower and upper
        lower_bds = np.array([bounds[idx][0] for idx in range(len(bounds))])
        upper_bds = np.array([bounds[idx][1] for idx in range(len(bounds))])

        # write all to file
        periods = np.arange(self.n_time_points)
        uncertainty_data = np.array([periods, lower_bds, self.lmp_sig_nom,
                                     upper_bds]).T
        np.savetxt('lmp_uncertainty_data.dat',
                   uncertainty_data, delimiter='\t',
                   header='time\tlower\tnominal\tupper',
                   comments='', fmt=['%d', '%.3f', '%.3f', '%.3f'])

    def pyros_set(self):
        """Obtain corresponding PyROS BoxSet."""
        return uncertainty_sets.BoxSet(bounds=self.bounds())

    def plot_bounds(self):
        """Plot LMP bounds against planning period."""
        bounds = self.bounds()

        # resolve bounds into lower and upper
        lower_bds = np.array([bounds[idx][0] for idx in range(len(bounds))])
        upper_bds = np.array([bounds[idx][1] for idx in range(len(bounds))])

        # generate the plots
        plt.plot(upper_bds, label='upper')
        plt.plot(self.lmp_sig_nom, label='nominal')
        plt.plot(lower_bds, label='lower')
        plt.xlabel('period')
        plt.ylabel('LMP signal')
        plt.legend()

        plt.show()


class HysterLMPBoxSet(SimpleLMPBoxSet):
    def __init__(self, lmp_data, n_recent, growth_rate, avg_multiplier,
                 latency):
        """Initialize hysteresis LMP box set."""
        super().__init__(lmp_data, n_recent, growth_rate, avg_multiplier)
        self.latency = latency

    def bounds(self):
        """Evaluate LMP uncertainty set bounds."""
        multipliers = [self.growth_rate * t for t in range(self.n_time_points)]

        # calculate moving averages
        moving_avgs = [self.mov_avg_multiplier
                       * sum([self.lmp_sig_nom[tprime]
                              / (t - max(0, t - self.n_recent))
                             for tprime in range(max(0, t - self.n_recent),
                                                 t - 1)])
                       for t in range(self.n_time_points)]

        # evaluate the bounds
        lmp_bounds = []
        drop_times = [time for time in range(self.n_time_points) if
                      self.lmp_sig_nom[time] == 0 and
                      self.lmp_sig_nom[max(0, time - 1)] > 0]
        spike_times = [time for time in range(self.n_time_points) if
                       self.lmp_sig_nom[time] > 0 and
                       self.lmp_sig_nom[max(0, time - 1)] == 0]
        spike_diffs = [self.lmp_sig_nom[time] - self.lmp_sig_nom[time - 1]
                       for time in spike_times]
        for time in range(self.n_time_points):
            # calculate lower bound
            time_low = max(0, time - self.latency)
            time_high = min(self.n_time_points, time + self.latency + 1)
            lb = max(0, self.lmp_sig_nom[time] * (1 - multipliers[time]))
            if np.any(self.lmp_sig_nom[time_low:time_high] == 0):
                lb = 0

            ub = min(200,
                     max(moving_avgs[time],
                         self.lmp_sig_nom[time] * (1 + multipliers[time])))
            if time in drop_times:
                diff = self.lmp_sig_nom[time - 1] - self.lmp_sig_nom[time - 2]
                ub = lmp_bounds[time - 1][1] + diff
            elif time + 1 in spike_times:
                diff = spike_diffs[spike_times == time]
                print(diff)
                ub = diff

            lmp_bounds.append((lb, ub))

        return lmp_bounds


if __name__ == "__main__":
    # decide set parameters
    n_periods = 41
    start = 0
    n_recent = 5
    growth_rate = 0.02
    mov_avg_mult = 0.2
    hyster_latency = 1

    lmp_data = get_lmp_data(n_periods, start=start)
    lmp_set = SimpleLMPBoxSet(lmp_data, n_recent, growth_rate, mov_avg_mult)
    # lmp_set.plot_bounds()

    hyster_set = HysterLMPBoxSet(lmp_data, n_recent, growth_rate, mov_avg_mult,
                                 hyster_latency)
    hyster_set.plot_bounds()
