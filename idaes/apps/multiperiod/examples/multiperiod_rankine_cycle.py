#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example driver for multiperiod.py using a simple rankine cycle.
"""

import os
from random import random

import numpy as np

import pyomo.environ as pyo
from pyomo.core.expr import current as EXPR
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables

from idaes.apps.rankine.simple_rankine_cycle import degrees_of_freedom
from idaes.core.util import to_json, from_json
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from idaes.apps.rankine.simple_rankine_cycle import (
    create_model,
    set_inputs,
    initialize_model,
    close_flowsheet_loop,
    add_operating_cost,
)


def is_equality(con):
    """
    Determine whether a constraint is an equality by
    checking whether bounds have the same value.
    """
    return (
        con.lower is not None
        and con.upper is not None
        and pyo.value(con.lower) == pyo.value(con.upper)
    )


def count_degrees_of_freedom(block, control_vars=None, print_results=False):
    """
    Count the number of equality constraints in a block
    and state variables participating in those constraints.
    """
    if control_vars is None:
        cvar_set = ComponentSet()
    else:
        cvar_set = ComponentSet(control_vars)

    cons = [
        con
        for con in block.component_data_objects(
            pyo.Constraint, active=True, descend_into=True
        )
        if is_equality(con)
    ]

    svars = ComponentSet(
        var
        for con in cons
        for var in identify_variables(con.body)
        if not var.fixed and var not in cvar_set
    )

    for var in control_vars:
        block.find_component(var.name).fix()

    if print_results:
        print("=" * 60)
        print("Num state variables:", len(svars))
        print("Num equality constraints:", len(cons))
        print("-" * 60)
        print("Degrees of freedom (this function):", len(svars) - len(cons))
        print("Degrees of freedom (idaes util):", degrees_of_freedom(block))
        print("=" * 60)

    for var in control_vars:
        block.find_component(var.name).unfix()

    return len(svars) - len(cons)


initialize_json_filename = "./initialized_state.json.gz"


def create_ss_rankine_model():
    """
    Create a steady-state model of the rankine cycle
    for a single time period block.
    """
    p_lower_bound = 300
    p_upper_bound = 450

    m = pyo.ConcreteModel()
    m.rankine = create_model(heat_recovery=True)
    m.rankine = set_inputs(m.rankine)

    # initialize the model
    if os.path.exists(initialize_json_filename):
        from_json(m.rankine, fname=initialize_json_filename, gz=True)
    else:
        m.rankine = initialize_model(m.rankine)
        to_json(
            m.rankine, fname=initialize_json_filename, gz=True, human_read=True
        )

    m.rankine = close_flowsheet_loop(m.rankine)
    m.rankine = add_operating_cost(m.rankine)

    # Setting bounds for net cycle power output for a capex design
    m.rankine.fs.eq_min_power = pyo.Constraint(
        expr=m.rankine.fs.net_cycle_power_output >= p_lower_bound * 1e6
    )

    m.rankine.fs.eq_max_power = pyo.Constraint(
        expr=m.rankine.fs.net_cycle_power_output <= p_upper_bound * 1e6
    )

    m.rankine.fs.boiler.inlet.flow_mol[0].unfix()
    m.rankine.fs.boiler.inlet.flow_mol[0].setlb(1)

    return m


def create_mp_rankine_block(lmp_signal):
    """
    Create the block for a single period of the multi-period
    rankine cycle model.

    Arguments
    ---------
        lmp_signal: price for time period
    """
    m = create_ss_rankine_model()
    b1 = m.rankine

    # Add new coupling variable for next time
    b1.power_output = pyo.Expression(expr=b1.fs.net_cycle_power_output * 1e-6)
    b1.next_power_output = pyo.Var(
        within=pyo.NonNegativeReals, bounds=(300, 450), initialize=1.5
    )

    # Add ramping constraint
    b1.ramp1 = pyo.Constraint(
        expr=b1.power_output - b1.next_power_output <= 100
    )
    b1.ramp2 = pyo.Constraint(
        expr=b1.next_power_output - b1.power_output <= 100
    )

    # Add battery integration
    b1.P_to_battery = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)
    b1.P_to_grid = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)
    b1.P_total = pyo.Constraint(
        expr=b1.power_output == b1.P_to_battery + b1.P_to_grid
    )

    # Simple battery model
    m.battery = pyo.Block()
    b2 = m.battery

    b2.soc = pyo.Var(
        within=pyo.NonNegativeReals, initialize=0.0, bounds=(0, 100)
    )
    b2.next_soc = pyo.Var(
        within=pyo.NonNegativeReals, initialize=0.0, bounds=(0, 100)
    )

    # Amount discharged to grid this time period (assume discharge is positive)
    b2.discharge = pyo.Var(initialize=0.0)
    b2.energy_change = pyo.Constraint(
        expr=b2.next_soc == b2.soc - b2.discharge + b1.P_to_battery
    )
    b2.energy_down_ramp = pyo.Constraint(expr=b2.soc - b2.next_soc <= 50)
    b2.energy_up_ramp = pyo.Constraint(expr=b2.next_soc - b2.soc <= 50)

    # Objective function
    m.lmp_signal = pyo.Param(initialize=lmp_signal, mutable=True)
    m.revenue = m.lmp_signal * (b1.P_to_grid + b2.discharge)
    m.cost = pyo.Expression(expr=-(m.revenue - b1.fs.operating_cost))
    return m


def get_rankine_link_variable_pairs(b1, b2):
    """
    Function to retrieve variable pairs linked between time steps.
    This is a user-provided function to a MultiPeriod class
        b1: current time block
        b2: next time block
    """
    return [
        (b1.rankine.next_power_output, b2.rankine.power_output),
        (b1.battery.next_soc, b2.battery.soc),
    ]


def get_vars_from_block(block):
    """
    Obtain all variables in a block.
    Returns an iterator.
    """
    seen = set()
    for var in block.component_data_objects(pyo.Var, sort=True):
        if id(var) not in seen:
            seen.add(id(var))
            yield var


def print_model_variables_by_block(par_block, fixed=None):
    """
    Print all variables participating in constraints for which
    the parent block is block.
    """
    print("MODEL VARIABLES:")
    for block in set(par_block.component_data_objects(pyo.Block)):
        if block.parent_block() is par_block:
            for v in get_vars_from_block(block):
                if type(fixed) is bool:
                    if v.fixed is fixed:
                        print(v.name)
                else:
                    print(v.name)

    print("-" * 60)


def create_mp_rankine_model(n_time_points, lmp_signal):
    """
    Create multiperiod rankine model with `n_time_points` hourly
    blocks and a given LMP signal (iterable of hourly prices).
    """
    lmp_signal = np.array(lmp_signal)
    assert lmp_signal.size == n_time_points

    mp_rankine = MultiPeriodModel(
        n_time_points, create_mp_rankine_block, get_rankine_link_variable_pairs
    )

    # data passed to each block time period
    time_points = np.arange(0, n_time_points)
    data_points = [
        {"lmp_signal": lmp_signal[idx]} for idx in range(n_time_points)
    ]
    data_kwargs = dict(zip(time_points, data_points))

    # create the multiperiod object
    mp_rankine.build_multi_period_model(data_kwargs)

    # remove json file used for ss rankine block initialization
    if os.path.exists(initialize_json_filename):
        os.remove(initialize_json_filename)

    return mp_rankine


if __name__ == "__main__":
    # ########################################
    #  check that a steady-state block solves
    # ########################################
    b_ss = create_ss_rankine_model()
    opt = pyo.SolverFactory("ipopt")
    opt.solve(b_ss)
    opt.solve(b_ss, tee=True)

    # ########################################
    #  Check that a multiperiod-capable block solves
    # ########################################
    b_mp = create_mp_rankine_block(lmp_signal=30.0)
    b_mp.obj = pyo.Objective(expr=b_mp.cost)
    b_mp.battery.soc.fix(0)
    opt = pyo.SolverFactory("ipopt")
    opt.solve(b_mp)
    opt.solve(b_mp, tee=True)

    # ########################################
    #  create and solve a multiperiod model
    # ########################################
    n_time_points = 7
    lmp_signal = np.array([random() * 100 for _ in range(n_time_points)])

    mp_rankine = create_mp_rankine_model(n_time_points, lmp_signal)

    #  retrieve pyomo model and active process blocks (i.e. time blocks)
    m = mp_rankine.pyomo_model
    blks = mp_rankine.get_active_process_blocks()
    # add_capex_plant(m)

    for blk in blks:
        print(blk.name)
    # get equations and variables for incidence analysis

    rankine_eqs = [
        c
        for c in m.component_data_objects(
            pyo.Constraint, active=True, descend_into=True
        )
        if c.equality
    ]

    rankine_vars = []
    seen = set()
    for eq in rankine_eqs:
        for var in EXPR.identify_variables(eq.expr, include_fixed=False):
            if id(var) not in seen and not var.fixed:
                seen.add(id(var))
                rankine_vars.append(var)

    # state_vars = [var for var in rankine_vars if ]
    print("Number equations:", len(rankine_eqs))
    print("Number variables:", len(rankine_vars))

    # print(m.capex_plant.name)

    print_model_variables_by_block(m.blocks[0])
    #  m.blocks[0].display()

    #  set objective function, fix initial state, and solve
    m.obj = pyo.Objective(expr=sum([blk.cost for blk in blks]))
    blks[0].battery.soc.fix(0)
    opt = pyo.SolverFactory("ipopt")
    opt.solve(m, tee=True)

    #  check results
    for block in blks:
        print(
            "Dispatch [MW]: ",
            pyo.value(block.rankine.power_output),
            "Charge [MWh]: ",
            pyo.value(block.battery.soc),
        )
    print_model_variables_by_block(m)
    m.display()
    for b in mp_rankine.get_active_process_blocks():
        print(b.name)
        b.rankine.display()

    # ########################################
    #  advance time steps -- i.e. deactivate first time period, and add new time
    #  period.
    # ########################################
    mp_rankine.advance_time(lmp_signal=0.0)
    blks = mp_rankine.get_active_process_blocks()

    #  Update data (objective price and battery storage level)
    m.obj.expr = sum([blk.cost for blk in blks])
    blks[0].battery.soc.fix(pyo.value(blks[0].battery.soc))
    blks[0].fix_power = pyo.Constraint(
        expr=blks[0].rankine.power_output
        == pyo.value(blks[0].rankine.power_output)
    )
    opt.solve(m, tee=True)

    # Check solution.
    for block in blks:
        print(
            "Dispatch [MW]: ",
            pyo.value(block.rankine.power_output),
            "Charge [MWh]: ",
            pyo.value(block.battery.soc),
        )

    # advance time again
    mp_rankine.advance_time(lmp_signal=200.0)
    blks = mp_rankine.get_active_process_blocks()
    m.obj.expr = sum([blk.cost for blk in blks])
    blks[0].battery.soc.fix(pyo.value(blks[0].battery.soc))
    blks[0].fix_power = pyo.Constraint(
        expr=blks[0].rankine.power_output
        == pyo.value(blks[0].rankine.power_output)
    )
    opt.solve(m, tee=True)
    for block in blks:
        print(
            "Dispatch [MW]: ",
            pyo.value(block.rankine.power_output),
            "Charge [MWh]: ",
            pyo.value(block.battery.soc),
        )
