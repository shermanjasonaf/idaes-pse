#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-11-11 Thu 05:18:42

@author: jasherma

Two-stage RO model of a multiperiod rankine cycle
"""

import multiperiod_rankine_cycle as mc
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import pyomo.contrib.pyros as pyros
from lmp_uncertainty_models import get_lmp_data, HysterLMPBoxSet


def plot_results(mp_model, lmp_signal):
    """
    Plot battery state of charge, rankine cycle power output
    as a function of the time period index and compare with LMP signal
    variations.
    """
    n_time_points = lmp_signal.size
    blks = mp_model.get_active_process_blocks()

    # get SOC and power output results from model
    soc_list = [pyo.value(blks[i].battery.soc) for i in range(n_time_points)]
    power_outputs = [pyo.value(blks[i].rankine.power_output)
                     for i in range(n_time_points)]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # plot state of charge
    ax1.plot(soc_list, 'r')
    ax1.set_xlabel('Time (hrs)')
    ax1.set_ylabel('State of Charge (MWh)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('LMP Signal ($/MWh)', color='black')
    ax2.plot(lmp_signal, 'k')
    plt.show()

    # plot power output
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.plot(power_outputs, 'r')
    ax1.set_xlabel('Time (hrs)')
    ax1.set_ylabel('Power Output (MWh)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('LMP Signal ($/MWh)', color='black')
    ax2.plot(lmp_signal, 'k')
    plt.show()


if __name__ == "__main__":
    # parameterization of uncertainty set
    start = 4000
    n_time_points = 7
    n_recent = 4
    box_growth_rate = 0.02
    moving_avg_multiplier = 0.1
    hyster_latency = 1

    # get LMP data
    lmp_signal = get_lmp_data(n_time_points, start=start)

    # construct container for uncertainty set
    lmp_set = HysterLMPBoxSet(lmp_signal, n_recent, box_growth_rate,
                              moving_avg_multiplier, hyster_latency,
                              start_day_hour=0, include_peak_effects=True)

    # create model
    mp_rankine = mc.create_mp_rankine_model(n_time_points, lmp_signal)

    # retrieve pyomo model and active process blocks (i.e. time blocks)
    m = mp_rankine.pyomo_model
    blks = mp_rankine.get_active_process_blocks()

    # set objective function, fix initial state, and solve deterministically
    m.obj = pyo.Objective(expr=sum([blk.cost for blk in blks]))
    blks[0].battery.soc.fix(0)
    # solver = pyo.SolverFactory('ipopt')
    # solver.solve(m, tee=False)

    # plot the results
    # plot_results(mp_rankine, lmp_signal)

    # check results
    for block in blks:
        print("Dispatch [MW]: ", pyo.value(block.rankine.power_output),
              "Charge [MWh]: ", pyo.value(block.battery.soc))

    # first and second-stage vars
    fs_vars = [blks[0].rankine.fs.boiler.inlet.flow_mol[0], 
               blks[0].rankine.P_to_grid, blks[0].battery.discharge]
    ss_vars = [blks[i].rankine.fs.boiler.inlet.flow_mol[0]
               for i in range(1, n_time_points)]
    ss_vars += [blks[i].rankine.P_to_grid for i in range(1, n_time_points)]
    ss_vars += [blks[i].battery.discharge for i in range(1, n_time_points)]

    # uncertain parameters
    uncertain_params = [blks[i].lmp_signal for i in range(n_time_points)]

    # solve with PyROS
    solve_pyros = True

    for con in m.component_data_objects(pyo.Constraint, active=True):
        if not con.equality:
            con.pprint()

    for expr in m.component_data_objects(pyo.Expression, active=True):
        print(expr.name)

    if solve_pyros:
        # solvers for PyROS
        local_solver = pyo.SolverFactory('ipopt')
        global_solver = pyo.SolverFactory('ipopt')
        pyros_solver = pyo.SolverFactory('pyros')

        results = pyros_solver.solve(
            model=mp_rankine.pyomo_model,
            first_stage_variables=fs_vars,
            second_stage_variables=ss_vars,
            uncertain_params=uncertain_params,
            uncertainty_set=lmp_set.pyros_set(),
            local_solver=local_solver,
            global_solver=global_solver,
            decision_rule_order=1,
            keepfiles=True,
            objective_focus=pyros.ObjectiveType.worst_case,
            tee=False,
            bypass_global_separation=True,
            subproblem_file_directory='./sublogs/'
        )
