#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-11-11 Thu 05:18:42

@author: jasherma

Two-stage robust optimization model of a multiperiod rankine cycle.
"""


import matplotlib.pyplot as plt

import pyomo.environ as pyo
import pyomo.contrib.pyros as pyros

from idaes.apps.multiperiod.examples.lmp_uncertainty_models import (
    get_lmp_data,
    HysterLMPBoxSet,
)
from idaes.apps.multiperiod.examples.multiperiod_rankine_cycle import (
    count_degrees_of_freedom,
)
import idaes.apps.multiperiod.examples.multiperiod_rankine_cycle as mc


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
    power_outputs = [
        pyo.value(blks[i].rankine.power_output) for i in range(n_time_points)
    ]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # plot state of charge
    ax1.plot(soc_list, "r")
    ax1.set_xlabel("Time (hrs)")
    ax1.set_ylabel("State of Charge (MWh)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("LMP Signal ($/MWh)", color="black")
    ax2.plot(lmp_signal, "k")
    plt.show()

    # plot power output
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.plot(power_outputs, "r")
    ax1.set_xlabel("Time (hrs)")
    ax1.set_ylabel("Power Output (MWh)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("LMP Signal ($/MWh)", color="black")
    ax2.plot(lmp_signal, "k")
    plt.show()


def create_two_stg_model(lmp_set, initialize_deterministic=True):
    """
    Create two-stage robust optimization model of a
    multi-period rankine cycle.

    Arguments
    ---------
    lmp_set: LMPBoxSet
        Object containing LMP data, and a method for
        deriving a PyROS UncertaintySet object.
    initialize_deterministic: bool, optional
        Initialize the two-stage model with the corresponding
        solution to the deterministic counterpart.
        The default is True.

    Returns
    -------
    mp_rankine : MultiPeriodModel
        The multiperiod model, with first-stage variables,
        second-stage variables, uncertain parameters,
        and uncertainty set embedded as attributes.
    """
    # create model
    mp_rankine = mc.create_mp_rankine_model(n_time_points, lmp_signal)

    # retrieve pyomo model and active process blocks (i.e. time blocks)
    m = mp_rankine.pyomo_model
    blks = mp_rankine.get_active_process_blocks()

    # set objective function, fix initial state,
    # and solve deterministic counterpart
    m.obj = pyo.Objective(expr=sum([blk.cost for blk in blks]))
    blks[0].battery.soc.fix(0)

    if initialize_deterministic:
        solver = pyo.SolverFactory('ipopt')
        solver.solve(m, tee=False)

    # plot the results
    # plot_results(mp_rankine, lmp_signal)

    # check results
    print("Printing results:")
    for block in blks:
        print(
            "Dispatch [MW]: ",
            pyo.value(block.rankine.power_output),
            "Charge [MWh]: ",
            pyo.value(block.battery.soc),
        )

    # first and second-stage vars
    fs_vars = [
        blks[0].rankine.fs.boiler.inlet.flow_mol[0],
        blks[0].rankine.P_to_grid,
        blks[0].battery.discharge,
    ]
    ss_vars = [
        blks[i].rankine.fs.boiler.inlet.flow_mol[0]
        for i in range(1, n_time_points)
    ]
    ss_vars += [blks[i].rankine.P_to_grid for i in range(1, n_time_points)]
    ss_vars += [blks[i].battery.discharge for i in range(1, n_time_points)]

    # uncertain parameters
    uncertain_params = [blks[i].lmp_signal for i in range(n_time_points)]

    # add attributes needed for PyROS
    mp_rankine.first_stage_variables = fs_vars
    mp_rankine.second_stage_variables = ss_vars
    mp_rankine.uncertain_params = uncertain_params
    mp_rankine.uncertainty_set = lmp_set.pyros_set()

    print(
        "Degrees of Freedom:",
        count_degrees_of_freedom(
            mp_rankine.pyomo_model, control_vars=fs_vars + ss_vars
        ),
    )

    return mp_rankine


if __name__ == "__main__":
    # parameterization of uncertainty set
    start = 4000
    n_time_points = 7
    n_recent = 4
    box_growth_rate = 0.02
    moving_avg_multiplier = 0.1
    hyster_latency = 1

    # solve two-stage model with PyROS?
    solve_pyros = True

    # get LMP data
    lmp_signal = get_lmp_data(n_time_points, start=start)

    # construct container for uncertainty set
    lmp_set = HysterLMPBoxSet(
        lmp_signal,
        n_recent,
        box_growth_rate,
        moving_avg_multiplier,
        hyster_latency,
        start_day_hour=0,
        include_peak_effects=True,
    )

    # create the two stage model
    # contains first-stage vars, uncertain params, and uncertainty
    # set as attributes
    mp_rankine = create_two_stg_model(lmp_set=lmp_set)

    if solve_pyros:
        # subsolvers. we use IPOPT as global since external fcns
        # make model nonamenable to global solvers
        local_solver = pyo.SolverFactory("ipopt")
        global_solver = pyo.SolverFactory("ipopt")

        # obtain PyROS solution
        pyros_solver = pyo.SolverFactory("pyros")
        results = pyros_solver.solve(
            model=mp_rankine.pyomo_model,
            first_stage_variables=mp_rankine.first_stage_variables,
            second_stage_variables=mp_rankine.second_stage_variables,
            uncertain_params=mp_rankine.uncertain_params,
            uncertainty_set=mp_rankine.uncertainty_set,
            local_solver=local_solver,
            global_solver=global_solver,
            decision_rule_order=0,
            keepfiles=True,
            objective_focus=pyros.ObjectiveType.worst_case,
            tee=False,
            bypass_global_separation=True,
            subproblem_file_directory="./sublogs/",
        )
