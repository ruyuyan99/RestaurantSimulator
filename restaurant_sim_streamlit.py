"""
Restaurant Simulation with Animation for Streamlit
==================================================

This Streamlit app provides an interactive interface to run and
visualise a discrete event simulation (DES) of a fast‑casual
restaurant.  In addition to numerical analysis and bar charts, this
version also produces an animated GIF showing customers walking
between ordering kiosks/registers, pickup, drink/condiment stations,
and seating.  Queues forming at each resource are displayed as
vertical bars growing and shrinking in real time.  Top counters show
how many registers, cooks and expo staff are busy relative to their
capacity.

The simulation logic is based on the salabim DES library for
queueing behaviour, but the animation is implemented from scratch
using matplotlib so it can run in a browser‑based Streamlit
environment without relying on Tkinter.

Usage
-----
Install the required packages (``salabim``, ``streamlit``, ``pandas``,
``matplotlib``, and ``numpy``) and run this script with Streamlit::

    streamlit run restaurant_sim_streamlit_animation.py

The sidebar allows you to customise the simulation horizon, arrival
rate, resource capacities and the proportion of customers who use
kiosks versus registers.  You can also specify how many minutes to
animate.  After running the simulation, summary metrics and charts
are displayed, followed by the animated GIF of the first part of the
run.
"""

import io
import random
import statistics
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
import streamlit as st
import salabim as sim


# ----------------------------------------------------------------------------
# Parameters and helper functions
# ----------------------------------------------------------------------------
# Default simulation horizon (hours)
DEFAULT_SIM_HOURS = 4

# Default arrival rate (customers per minute)
DEFAULT_ARRIVAL_RATE = 1.2

# Default party size distribution
PARTY_SIZE_WEIGHTS: List[Tuple[int, float]] = [
    (1, 0.25),
    (2, 0.45),
    (3, 0.15),
    (4, 0.15),
]

# Walking speed (m/s)
WALK_SPEED_MPS = 1.2

# Distances between nodes (m).  These are used to compute walking times.
# Nodes: DOOR, KIOSK, REGISTER, PICKUP, DRINK, CONDIMENT, SEATING, EXIT
DISTANCES = [
    [0, 12, 14, 20, 24, 26, 30, 36],  # DOOR ->
    [12, 0,  6, 14, 10, 12, 18, 26],  # KIOSK
    [14, 6,  0, 12, 10, 12, 18, 26],  # REGISTER
    [20,14, 12, 0,  8,  10, 12, 20],  # PICKUP
    [24,10, 10, 8,  0,  6,  10, 18],  # DRINK
    [26,12, 12,10, 6,  0,  10, 18],   # CONDIMENT
    [30,18, 18,12,10, 10,  0,  12],   # SEATING
    [36,26, 26,20,18, 18, 12,  0],    # EXIT
]

# Node indices for readability
DOOR, KIOSK, REGISTER, PICKUP, DRINK, CONDIMENT, SEATING, EXIT = range(8)

# Mean service times (seconds)
MEAN_ORDER_TIME     = 55  # kiosk
MEAN_REGISTER_TIME  = 65  # register
MEAN_COOK_TIME      = 210
MEAN_EXPO_TIME      = 20
MEAN_DRINK_TIME     = 12
MEAN_CONDIMENT_TIME = 10
MEAN_DINE_TIME_MIN  = {1: 14, 2: 20, 3: 24, 4: 28}  # dine time (minutes) by party size


def walk_time(a: int, b: int) -> float:
    """Return the walking time (seconds) between nodes a and b.

    A small random congestion factor (1.0–1.25) is applied to the base
    time, which is the distance divided by walking speed.  This mirrors
    the behaviour used in the original animation script.
    """
    base = DISTANCES[a][b] / WALK_SPEED_MPS
    return base * random.uniform(1.0, 1.25)


def expov(mean: float) -> float:
    """Exponential draw with a floor of 1 second to avoid extremely small samples."""
    return max(1.0, random.expovariate(1 / mean))


def party_size() -> int:
    """Draw a party size based on PARTY_SIZE_WEIGHTS."""
    r = random.random()
    acc = 0.0
    for size, weight in PARTY_SIZE_WEIGHTS:
        acc += weight
        if r <= acc:
            return size
    return PARTY_SIZE_WEIGHTS[-1][0]


# ----------------------------------------------------------------------------
# Salabim simulation components (for summary statistics)
# ----------------------------------------------------------------------------
class Restaurant(sim.Component):
    """Container for resources and performance data."""

    def setup(
        self,
        n_kiosks: int,
        n_registers: int,
        n_cooks: int,
        n_expo: int,
        n_drinks: int,
        n_condiments: int,
        n_tables: int,
    ):
        # Resources
        self.kiosks     = sim.Resource("kiosks", capacity=n_kiosks)
        self.registers  = sim.Resource("registers", capacity=n_registers)
        self.cooks      = sim.Resource("cooks", capacity=n_cooks)
        self.expo       = sim.Resource("expo", capacity=n_expo)
        self.drinks     = sim.Resource("drinks", capacity=n_drinks)
        self.condiments = sim.Resource("condiments", capacity=n_condiments)
        self.tables     = sim.Resource("tables", capacity=n_tables)

        # Performance data
        self.completed_times: List[float] = []


class Customer(sim.Component):
    """Simulates a customer moving through the restaurant without animation."""

    def setup(self, rid: int, restaurant: Restaurant, pct_to_kiosk: float):
        self.rid = rid
        self.r = restaurant
        self.pct_to_kiosk = pct_to_kiosk
        self.party = party_size()
        self.t_start = self.env.now()

    def process(self):
        # Choose kiosk or register
        to_kiosk = (random.random() < self.pct_to_kiosk) and (self.r.kiosks.capacity() > 0)
        first = KIOSK if to_kiosk else REGISTER

        # Walk from door to order point
        dur = walk_time(DOOR, first)
        yield self.hold(dur)

        # Order at kiosk or register
        if to_kiosk:
            svc = expov(MEAN_ORDER_TIME)
            req = self.request(self.r.kiosks)
            yield req
            yield self.hold(svc)
            self.release(self.r.kiosks)
        else:
            svc = expov(MEAN_REGISTER_TIME)
            req = self.request(self.r.registers)
            yield req
            yield self.hold(svc)
            self.release(self.r.registers)

        # Walk to pickup
        dur = walk_time(first, PICKUP)
        yield self.hold(dur)

        # Kitchen
        svc = expov(MEAN_COOK_TIME)
        req = self.request(self.r.cooks)
        yield req
        yield self.hold(svc)
        self.release(self.r.cooks)

        # Expo
        svc = expov(MEAN_EXPO_TIME)
        req = self.request(self.r.expo)
        yield req
        yield self.hold(svc)
        self.release(self.r.expo)

        # Walk to drinks
        dur = walk_time(PICKUP, DRINK)
        yield self.hold(dur)

        # Drinks
        svc = expov(MEAN_DRINK_TIME)
        req = self.request(self.r.drinks)
        yield req
        yield self.hold(svc)
        self.release(self.r.drinks)

        # Walk to condiments
        dur = walk_time(DRINK, CONDIMENT)
        yield self.hold(dur)

        # Condiments
        svc = expov(MEAN_CONDIMENT_TIME)
        req = self.request(self.r.condiments)
        yield req
        yield self.hold(svc)
        self.release(self.r.condiments)

        # Walk to seating
        dur = walk_time(CONDIMENT, SEATING)
        yield self.hold(dur)

        # Seating
        dine_mean_min = (
            MEAN_DINE_TIME_MIN[self.party]
            if isinstance(MEAN_DINE_TIME_MIN, dict)
            else MEAN_DINE_TIME_MIN
        )
        dine_t = 60.0 * expov(dine_mean_min)
        req = self.request(self.r.tables)
        yield req
        yield self.hold(dine_t)
        self.release(self.r.tables)

        # Walk to exit
        dur = walk_time(SEATING, EXIT)
        yield self.hold(dur)

        # Record total time in system
        total = self.env.now() - self.t_start
        self.r.completed_times.append(total)


class Arrivals(sim.Component):
    """Generates customers with exponential interarrival times."""

    def setup(self, restaurant: Restaurant, sim_hours: float, arrival_rate: float, pct_to_kiosk: float):
        self.r = restaurant
        self.sim_seconds = sim_hours * 3600
        self.arrival_rate = arrival_rate  # per minute
        self.pct_to_kiosk = pct_to_kiosk

    def process(self):
        while self.env.now() < self.sim_seconds:
            interarrival = random.expovariate(self.arrival_rate) * 60.0
            yield self.hold(interarrival)
            Customer(restaurant=self.r, rid=self.env.now(), pct_to_kiosk=self.pct_to_kiosk)


def run_simulation(
    sim_hours: float,
    arrival_rate: float,
    n_kiosks: int,
    n_registers: int,
    n_cooks: int,
    n_expo: int,
    n_drinks: int,
    n_condiments: int,
    table_cap: int,
    pct_to_kiosk: float,
) -> Dict[str, float]:
    """Run a single replication of the restaurant simulation and return metrics.

    This function executes the simulation using salabim without animation and
    collects summary statistics and resource utilization data.

    Parameters
    ----------
    sim_hours : float
        Length of the simulation horizon in hours.
    arrival_rate : float
        Customer arrival rate (customers per minute).
    n_kiosks, n_registers, n_cooks, n_expo, n_drinks, n_condiments : int
        Capacities of the corresponding resources.
    table_cap : int
        Total number of seats available (sum over table sizes).
    pct_to_kiosk : float
        Fraction of customers who choose kiosks over registers.

    Returns
    -------
    dict
        A dictionary containing metrics including number served, average and
        percentile times in system, and utilization per resource.
    """
    # Create a new environment
    sim.yieldless(False)  # allow yield statements
    env = sim.Environment(trace=False)

    # Instantiate the restaurant and arrival process
    R = Restaurant(
        n_kiosks=n_kiosks,
        n_registers=n_registers,
        n_cooks=n_cooks,
        n_expo=n_expo,
        n_drinks=n_drinks,
        n_condiments=n_condiments,
        n_tables=table_cap,
    )
    Arrivals(restaurant=R, sim_hours=sim_hours, arrival_rate=arrival_rate, pct_to_kiosk=pct_to_kiosk)

    # Run the simulation
    env.run(till=sim_hours * 3600)

    # Collect metrics
    results: Dict[str, float] = {}
    served = len(R.completed_times)
    results["served"] = served
    if served > 0:
        avg_time = statistics.mean(R.completed_times) / 60.0  # convert to minutes
        results["avg_time_min"] = avg_time
        results["p90_time_min"] = statistics.quantiles(R.completed_times, n=10)[-1] / 60.0 if served >= 10 else float('nan')
    else:
        results["avg_time_min"] = float('nan')
        results["p90_time_min"] = float('nan')

    # Utilization: average number of busy servers divided by capacity
    def util(res: sim.Resource) -> float:
        mean_claimers = res.claimers().length.mean()
        cap = res.capacity()
        return mean_claimers / cap if cap > 0 else 0.0

    results["util_kiosks"] = util(R.kiosks)
    results["util_registers"] = util(R.registers)
    results["util_cooks"] = util(R.cooks)
    results["util_expo"] = util(R.expo)
    results["util_drinks"] = util(R.drinks)
    results["util_condiments"] = util(R.condiments)
    results["util_tables"] = util(R.tables)

    # Mean queue lengths
    def qlen_mean(res: sim.Resource) -> float:
        return res.requesters().length.mean()

    results["qlen_kiosks"] = qlen_mean(R.kiosks)
    results["qlen_registers"] = qlen_mean(R.registers)
    results["qlen_cooks"] = qlen_mean(R.cooks)
    results["qlen_expo"] = qlen_mean(R.expo)
    results["qlen_drinks"] = qlen_mean(R.drinks)
    results["qlen_condiments"] = qlen_mean(R.condiments)
    results["qlen_tables"] = qlen_mean(R.tables)

    return results


# ----------------------------------------------------------------------------
# Custom simulation for animation
# ----------------------------------------------------------------------------
class ResourceState:
    """Internal helper class to track resource servers, queue and busy events."""

    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self.free_times: List[float] = [0.0] * capacity  # times when each server is free
        self.queue_events: List[Tuple[float, int]] = []  # (time, +1 or -1) when someone joins/leaves queue
        self.busy_events: List[Tuple[float, int]] = []   # (time, +1 or -1) when server starts/ends service


def schedule_service(res_state: ResourceState, arrival_time: float, mean_service: float) -> Tuple[float, float]:
    """Schedule a service at a resource and record queue/busy events.

    Parameters
    ----------
    res_state : ResourceState
        The resource for which the service is scheduled.
    arrival_time : float
        The time when the customer arrives to request service.
    mean_service : float
        The mean of the exponential service time (seconds).

    Returns
    -------
    (start_service_time, end_service_time)
    """
    if res_state.capacity <= 0:
        # No capacity at this resource; service happens immediately with zero duration
        return arrival_time, arrival_time

    # Find the earliest available server
    idx = int(np.argmin(res_state.free_times))
    earliest_free = res_state.free_times[idx]

    # The service cannot start before the customer arrives
    start_service = max(arrival_time, earliest_free)
    # Determine if customer had to wait
    if start_service > arrival_time:
        # Customer waits in queue until start_service
        res_state.queue_events.append((arrival_time, +1))
        res_state.queue_events.append((start_service, -1))
    else:
        # Customer does not wait; queue length does not change (but record zero wait as +0)
        res_state.queue_events.append((arrival_time, 0))
    # Schedule service duration
    dur = expov(mean_service)
    end_service = start_service + dur
    # Record busy events
    res_state.busy_events.append((start_service, +1))
    res_state.busy_events.append((end_service, -1))
    # Update the server's next free time
    res_state.free_times[idx] = end_service
    return start_service, end_service


def simulate_customers(
    sim_hours: float,
    arrival_rate: float,
    capacities: Dict[str, int],
    pct_to_kiosk: float,
    animation_seconds: float,
) -> Tuple[List[List[Dict[str, float]]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run a custom DES to generate movement segments and queue/busy time series for animation.

    This function simulates customer arrivals and service without using
    salabim.  It returns a list of per‑customer segment lists, where
    each segment contains start and end times and positions, and a
    dictionary of queue lengths and busy fractions for each resource
    sampled at a 1‑second resolution up to ``animation_seconds``.

    Parameters
    ----------
    sim_hours : float
        Total duration of the simulation (for generating arrivals), in hours.
    arrival_rate : float
        Customer arrival rate (customers per minute).
    capacities : dict
        Dictionary mapping resource names to capacities.  Keys must
        include 'kiosks', 'registers', 'cooks', 'expo', 'drinks',
        'condiments', and 'tables'.
    pct_to_kiosk : float
        Fraction of customers who choose kiosks over registers.
    animation_seconds : float
        Duration (seconds) to simulate and animate.

    Returns
    -------
    segments : List[List[Dict]]
        A list of lists; each sublist contains segment dictionaries for one
        customer.  Each segment dictionary has keys ``start_time``,
        ``end_time``, ``x0``, ``y0``, ``x1``, ``y1``, and ``wait``
        indicating whether the segment represents waiting/service at a
        station (True) or a walk (False).
    queue_ts : Dict[str, numpy.ndarray]
        For each resource name, a 1‑D array of queue lengths sampled
        every second (length = ``int(animation_seconds) + 1``).
    busy_ts : Dict[str, numpy.ndarray]
        For each resource name, a 1‑D array of busy server counts
        sampled every second.
    """
    # Define node coordinates (based on earlier animation positions) and normalise to [0, 10] x [0, 4]
    raw_pos = {
        DOOR:      (50, 250),
        KIOSK:     (250, 120),
        REGISTER:  (250, 380),
        PICKUP:    (480, 250),
        DRINK:     (680, 200),
        CONDIMENT: (820, 220),
        SEATING:   (1050, 260),
        EXIT:      (1250, 260),
    }
    xs = [p[0] for p in raw_pos.values()]
    ys = [p[1] for p in raw_pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Normalise positions so they fit nicely in a [0, 10] x [0, 4] box
    def norm(x: float, y: float) -> Tuple[float, float]:
        return ((x - min_x) / (max_x - min_x) * 10.0,
                (y - min_y) / (max_y - min_y) * 4.0)
    NODE_COORDS = {node: norm(*pos) for node, pos in raw_pos.items()}

    # Create resource states
    resources: Dict[str, ResourceState] = {
        'kiosks':     ResourceState('kiosks',     capacities.get('kiosks', 0)),
        'registers':  ResourceState('registers',  capacities.get('registers', 0)),
        'cooks':      ResourceState('cooks',      capacities.get('cooks', 0)),
        'expo':       ResourceState('expo',       capacities.get('expo', 0)),
        'drinks':     ResourceState('drinks',     capacities.get('drinks', 0)),
        'condiments': ResourceState('condiments', capacities.get('condiments', 0)),
        'tables':     ResourceState('tables',     capacities.get('tables', 0)),
    }

    # Generate arrival times
    arrival_times: List[float] = []
    t = 0.0
    while t < sim_hours * 3600:
        inter = random.expovariate(arrival_rate) * 60.0
        t += inter
        if t > sim_hours * 3600:
            break
        arrival_times.append(t)

    segments: List[List[Dict[str, float]]] = []

    for arrival_time in arrival_times:
        # Stop generating segments after the animation window
        if arrival_time > animation_seconds:
            break
        customer_segments: List[Dict[str, float]] = []
        current_time = arrival_time
        current_node = DOOR

        # Draw party size and decide kiosk vs register
        party = party_size()
        to_kiosk = (random.random() < pct_to_kiosk) and (capacities.get('kiosks', 0) > 0)
        first = KIOSK if to_kiosk else REGISTER

        # Walk to first station
        walk_dur = walk_time(current_node, first)
        seg = {
            'start_time': current_time,
            'end_time': current_time + walk_dur,
            'x0': NODE_COORDS[current_node][0],
            'y0': NODE_COORDS[current_node][1],
            'x1': NODE_COORDS[first][0],
            'y1': NODE_COORDS[first][1],
            'wait': False,
        }
        customer_segments.append(seg)
        current_time += walk_dur
        current_node = first

        # Service at kiosk/register
        if to_kiosk:
            start_svc, end_svc = schedule_service(resources['kiosks'], current_time, MEAN_ORDER_TIME)
        else:
            start_svc, end_svc = schedule_service(resources['registers'], current_time, MEAN_REGISTER_TIME)
        # Wait segment if any
        if start_svc > current_time:
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        # Service segment
        if end_svc > start_svc:
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        current_time = end_svc

        # Walk to pickup
        walk_dur = walk_time(current_node, PICKUP)
        customer_segments.append({
            'start_time': current_time,
            'end_time': current_time + walk_dur,
            'x0': NODE_COORDS[current_node][0],
            'y0': NODE_COORDS[current_node][1],
            'x1': NODE_COORDS[PICKUP][0],
            'y1': NODE_COORDS[PICKUP][1],
            'wait': False,
        })
        current_time += walk_dur
        current_node = PICKUP

        # Service at cook
        start_svc, end_svc = schedule_service(resources['cooks'], current_time, MEAN_COOK_TIME)
        if start_svc > current_time:
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        if end_svc > start_svc:
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        current_time = end_svc

        # Service at expo
        start_svc, end_svc = schedule_service(resources['expo'], current_time, MEAN_EXPO_TIME)
        if start_svc > current_time:
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        if end_svc > start_svc:
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        current_time = end_svc

        # Walk to drinks
        walk_dur = walk_time(current_node, DRINK)
        customer_segments.append({
            'start_time': current_time,
            'end_time': current_time + walk_dur,
            'x0': NODE_COORDS[current_node][0],
            'y0': NODE_COORDS[current_node][1],
            'x1': NODE_COORDS[DRINK][0],
            'y1': NODE_COORDS[DRINK][1],
            'wait': False,
        })
        current_time += walk_dur
        current_node = DRINK

        # Service at drinks
        start_svc, end_svc = schedule_service(resources['drinks'], current_time, MEAN_DRINK_TIME)
        if start_svc > current_time:
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        if end_svc > start_svc:
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        current_time = end_svc

        # Walk to condiments
        walk_dur = walk_time(current_node, CONDIMENT)
        customer_segments.append({
            'start_time': current_time,
            'end_time': current_time + walk_dur,
            'x0': NODE_COORDS[current_node][0],
            'y0': NODE_COORDS[current_node][1],
            'x1': NODE_COORDS[CONDIMENT][0],
            'y1': NODE_COORDS[CONDIMENT][1],
            'wait': False,
        })
        current_time += walk_dur
        current_node = CONDIMENT

        # Service at condiments
        start_svc, end_svc = schedule_service(resources['condiments'], current_time, MEAN_CONDIMENT_TIME)
        if start_svc > current_time:
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        if end_svc > start_svc:
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        current_time = end_svc

        # Walk to seating
        walk_dur = walk_time(current_node, SEATING)
        customer_segments.append({
            'start_time': current_time,
            'end_time': current_time + walk_dur,
            'x0': NODE_COORDS[current_node][0],
            'y0': NODE_COORDS[current_node][1],
            'x1': NODE_COORDS[SEATING][0],
            'y1': NODE_COORDS[SEATING][1],
            'wait': False,
        })
        current_time += walk_dur
        current_node = SEATING

        # Seating (tables)
        dine_mean_min = (
            MEAN_DINE_TIME_MIN[party]
            if isinstance(MEAN_DINE_TIME_MIN, dict)
            else MEAN_DINE_TIME_MIN
        )
        dine_t = 60.0 * expov(dine_mean_min)
        start_svc, end_svc = schedule_service(resources['tables'], current_time, dine_t)
        # Wait and dine segments
        if start_svc > current_time:
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        if end_svc > start_svc:
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': NODE_COORDS[current_node][0],
                'y0': NODE_COORDS[current_node][1],
                'x1': NODE_COORDS[current_node][0],
                'y1': NODE_COORDS[current_node][1],
                'wait': True,
            })
        current_time = end_svc

        # Walk to exit
        walk_dur = walk_time(current_node, EXIT)
        customer_segments.append({
            'start_time': current_time,
            'end_time': current_time + walk_dur,
            'x0': NODE_COORDS[current_node][0],
            'y0': NODE_COORDS[current_node][1],
            'x1': NODE_COORDS[EXIT][0],
            'y1': NODE_COORDS[EXIT][1],
            'wait': False,
        })
        # Append to list of customers
        segments.append(customer_segments)

    # Determine the length of the time series (sampled every second)
    horizon = int(animation_seconds) + 1
    queue_ts: Dict[str, np.ndarray] = {}
    busy_ts: Dict[str, np.ndarray] = {}
    times = np.arange(horizon)

    # Compute queue length time series for each resource
    for key, res_state in resources.items():
        events = sorted(res_state.queue_events, key=lambda x: x[0])
        q = 0
        idx = 0
        arr = np.zeros(horizon)
        for i, t_sec in enumerate(times):
            while idx < len(events) and events[idx][0] <= t_sec:
                q += events[idx][1]
                idx += 1
            arr[i] = q
        queue_ts[key] = arr

    # Compute busy servers time series for each resource
    for key, res_state in resources.items():
        events = sorted(res_state.busy_events, key=lambda x: x[0])
        b = 0
        idx = 0
        arr = np.zeros(horizon)
        for i, t_sec in enumerate(times):
            while idx < len(events) and events[idx][0] <= t_sec:
                b += events[idx][1]
                idx += 1
            arr[i] = b
        busy_ts[key] = arr

    return segments, queue_ts, busy_ts


def create_animation_gif(
    segments: List[List[Dict[str, float]]],
    queue_ts: Dict[str, np.ndarray],
    busy_ts: Dict[str, np.ndarray],
    animation_seconds: float,
    capacities: Dict[str, int],
    filename: str = 'animation.gif',
    fps: int = 10,
) -> bytes:
    """Create an animated GIF from simulation segments and queue/busy time series.

    Parameters
    ----------
    segments : list
        Per‑customer segment data returned by ``simulate_customers``.
    queue_ts : dict
        Queue length time series by resource.
    busy_ts : dict
        Busy server counts time series by resource.
    animation_seconds : float
        Duration of the animation in seconds.
    capacities : dict
        Resource capacities used to compute busy fractions.
    filename : str, optional
        Name for the output GIF file.
    fps : int, optional
        Frames per second in the animation.

    Returns
    -------
    bytes
        Bytes of the generated GIF.
    """
    # Determine time steps; we sample at frame_interval seconds
    frame_interval = 1.0 / fps
    num_frames = int(animation_seconds / frame_interval) + 1
    times = np.linspace(0.0, animation_seconds, num_frames)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Restaurant Simulation Animation')
    # Draw static nodes as rectangles with labels
    node_coords = {
        DOOR:      (50, 250),
        KIOSK:     (250, 120),
        REGISTER:  (250, 380),
        PICKUP:    (480, 250),
        DRINK:     (680, 200),
        CONDIMENT: (820, 220),
        SEATING:   (1050, 260),
        EXIT:      (1250, 260),
    }
    xs = [p[0] for p in node_coords.values()]
    ys = [p[1] for p in node_coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    def norm(x: float, y: float) -> Tuple[float, float]:
        return ((x - min_x) / (max_x - min_x) * 10.0,
                (y - min_y) / (max_y - min_y) * 4.0)
    NODE_COORDS = {node: norm(*pos) for node, pos in node_coords.items()}

    node_labels = {
        DOOR: 'Door',
        KIOSK: 'Kiosk',
        REGISTER: 'Register',
        PICKUP: 'Pickup',
        DRINK: 'Drink',
        CONDIMENT: 'Condiment',
        SEATING: 'Seating',
        EXIT: 'Exit',
    }
    # Draw node rectangles
    for node, (x, y) in NODE_COORDS.items():
        rect = Rectangle((x - 0.3, y - 0.2), 0.6, 0.4, facecolor='lightgray', edgecolor='gray')
        ax.add_patch(rect)
        ax.text(x, y, node_labels[node], ha='center', va='center', fontsize=8)

    # Prepare dynamic artists
    # Initialise scatter for customers.  Use an empty Nx2 array to avoid
    # indexing errors in set_offsets when there are no customers yet.
    scat = ax.scatter([], [], s=30, c='blue')
    # Queue bars for selected resources: display bars for kiosk, register, cooks, expo, drinks, condiments, tables
    bar_artists: Dict[str, Rectangle] = {}
    bar_positions = {
        'kiosks':     (NODE_COORDS[KIOSK][0], NODE_COORDS[KIOSK][1] + 0.3),
        'registers':  (NODE_COORDS[REGISTER][0], NODE_COORDS[REGISTER][1] + 0.3),
        'cooks':      (NODE_COORDS[PICKUP][0] - 0.4, NODE_COORDS[PICKUP][1] + 0.3),
        'expo':       (NODE_COORDS[PICKUP][0] + 0.4, NODE_COORDS[PICKUP][1] + 0.3),
        'drinks':     (NODE_COORDS[DRINK][0], NODE_COORDS[DRINK][1] + 0.3),
        'condiments': (NODE_COORDS[CONDIMENT][0], NODE_COORDS[CONDIMENT][1] + 0.3),
        'tables':     (NODE_COORDS[SEATING][0], NODE_COORDS[SEATING][1] + 0.3),
    }
    for key, (bx, by) in bar_positions.items():
        bar = Rectangle((bx - 0.05, by), 0.1, 0.0, facecolor='darkgray', edgecolor='none')
        ax.add_patch(bar)
        bar_artists[key] = bar
    # Text for queue lengths
    queue_texts: Dict[str, any] = {}
    for key, (bx, by) in bar_positions.items():
        qt = ax.text(bx, by + 0.05, '', ha='center', va='bottom', fontsize=6)
        queue_texts[key] = qt
    # Top counters for busy fractions (registers, cooks, expo)
    top_text = ax.text(0.5, 3.8, '', ha='left', va='center', fontsize=8)

    # Flatten segments into a list of segments with pointers to current segment index per customer
    customer_states = []
    for cust_segments in segments:
        if cust_segments:
            customer_states.append({'segments': cust_segments, 'index': 0})

    # Precompute queue lengths and busy counts at each frame time
    def get_queue_length(res_key: str, t: float) -> float:
        idx = int(min(max(int(t), 0), len(queue_ts[res_key]) - 1))
        return queue_ts[res_key][idx]
    def get_busy_count(res_key: str, t: float) -> float:
        idx = int(min(max(int(t), 0), len(busy_ts[res_key]) - 1))
        return busy_ts[res_key][idx]

    def init():
        # When no positions exist, set offsets to an empty array of shape (0, 2)
        scat.set_offsets(np.empty((0, 2)))
        for key in bar_artists:
            bar_artists[key].set_height(0.0)
            queue_texts[key].set_text('')
        top_text.set_text('')
        return [scat, *bar_artists.values(), *queue_texts.values(), top_text]

    def update(frame: int):
        t = times[frame]
        # Update positions
        pos_list: List[Tuple[float, float]] = []
        for state in customer_states:
            segs = state['segments']
            i = state['index']
            # Advance segment index if necessary
            while i < len(segs) and t >= segs[i]['end_time']:
                i += 1
            state['index'] = i
            if i < len(segs) and segs[i]['start_time'] <= t < segs[i]['end_time']:
                seg = segs[i]
                if seg['wait'] or seg['end_time'] == seg['start_time']:
                    # stay at fixed position
                    pos_list.append((seg['x0'], seg['y0']))
                else:
                    frac = (t - seg['start_time']) / (seg['end_time'] - seg['start_time']) if seg['end_time'] > seg['start_time'] else 0.0
                    x = seg['x0'] + frac * (seg['x1'] - seg['x0'])
                    y = seg['y0'] + frac * (seg['y1'] - seg['y0'])
                    pos_list.append((x, y))
        scat.set_offsets(pos_list)

        # Update queue bars and queue length text
        for key in bar_artists:
            qlen = get_queue_length(key, t)
            height = 0.05 * qlen  # scale factor for bar height
            bar_artists[key].set_height(height)
            queue_texts[key].set_text(f'Q: {int(qlen)}')
        # Update top counters for busy resources
        b_regs = get_busy_count('registers', t)
        b_cooks = get_busy_count('cooks', t)
        b_expo  = get_busy_count('expo', t)
        cap_regs = capacities.get('registers', 1)
        cap_cooks = capacities.get('cooks', 1)
        cap_expo  = capacities.get('expo', 1)
        top_text.set_text(
            f'Registers: {int(b_regs)}/{cap_regs}   '
            f'Cooks: {int(b_cooks)}/{cap_cooks}   '
            f'Expo: {int(b_expo)}/{cap_expo}'
        )
        return [scat, *bar_artists.values(), *queue_texts.values(), top_text]

    # Use blit=False to avoid backend issues in headless environments such as
    # Streamlit Cloud.  Blitting can cause crashes when scatter offsets are empty.
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)
    # Save to GIF into a BytesIO buffer
    buf = io.BytesIO()
    writer = PillowWriter(fps=fps)
    anim.save(buf, writer=writer)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ----------------------------------------------------------------------------
# Streamlit interface
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Restaurant DES with Animation", layout="wide")
    st.title("🍽️ Restaurant Discrete Event Simulation with Animation")
    st.write(
        "This interactive app simulates a fast casual restaurant. Adjust the\n"
        "parameters in the sidebar and click **Run Simulation** to see how\n"
        "changes affect throughput, waiting times, and resource utilization.\n"
        "You can also generate an animated GIF showing the first part of the run."
    )

    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    sim_hours = st.sidebar.slider("Simulation duration (hours)", 1, 12, DEFAULT_SIM_HOURS)
    arrival_rate = st.sidebar.slider("Arrival rate (customers per minute)", 0.1, 5.0, DEFAULT_ARRIVAL_RATE, 0.1)
    pct_to_kiosk = st.sidebar.slider("Fraction choosing kiosk", 0.0, 1.0, 0.75, 0.05)

    st.sidebar.subheader("Resource capacities")
    n_kiosks = st.sidebar.number_input("Number of kiosks", min_value=0, max_value=20, value=6, step=1)
    n_registers = st.sidebar.number_input("Number of registers", min_value=0, max_value=20, value=2, step=1)
    n_cooks = st.sidebar.number_input("Number of cooks", min_value=0, max_value=20, value=5, step=1)
    n_expo = st.sidebar.number_input("Number of expo staff", min_value=0, max_value=20, value=1, step=1)
    n_drinks = st.sidebar.number_input("Number of drink stations", min_value=0, max_value=20, value=2, step=1)
    n_condiments = st.sidebar.number_input("Number of condiment stations", min_value=0, max_value=20, value=2, step=1)

    st.sidebar.subheader("Seating")
    table_cap = st.sidebar.number_input(
        "Total seats (sum over tables)", min_value=0, max_value=200, value=sum({2: 18, 4: 10}.values()) * 2, step=2
    )

    st.sidebar.subheader("Animation")
    anim_minutes = st.sidebar.slider("Animate first X minutes", 1, 60, 10)
    fps = st.sidebar.select_slider("Animation FPS", options=[5, 10, 15], value=10)

    run_button = st.sidebar.button("Run Simulation and Animate")

    if run_button:
        with st.spinner("Running simulation and generating animation, please wait..."):
            # Run the salabim simulation for full horizon to collect stats
            results = run_simulation(
                sim_hours=sim_hours,
                arrival_rate=arrival_rate,
                n_kiosks=int(n_kiosks),
                n_registers=int(n_registers),
                n_cooks=int(n_cooks),
                n_expo=int(n_expo),
                n_drinks=int(n_drinks),
                n_condiments=int(n_condiments),
                table_cap=int(table_cap),
                pct_to_kiosk=pct_to_kiosk,
            )
            # Prepare capacities dictionary for custom simulation
            caps = {
                'kiosks': int(n_kiosks),
                'registers': int(n_registers),
                'cooks': int(n_cooks),
                'expo': int(n_expo),
                'drinks': int(n_drinks),
                'condiments': int(n_condiments),
                'tables': int(table_cap),
            }
            anim_seconds = anim_minutes * 60.0
            # Run custom simulation for the animation horizon
            segs, qts, bts = simulate_customers(
                sim_hours=sim_hours,
                arrival_rate=arrival_rate,
                capacities=caps,
                pct_to_kiosk=pct_to_kiosk,
                animation_seconds=anim_seconds,
            )
            # Create GIF
            gif_bytes = create_animation_gif(
                segments=segs,
                queue_ts=qts,
                busy_ts=bts,
                animation_seconds=anim_seconds,
                capacities=caps,
                filename='restaurant_animation.gif',
                fps=int(fps),
            )
        # Display summary metrics
        st.subheader("Summary Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers served", results["served"])
        c2.metric("Avg time in system (min)", f"{results['avg_time_min']:.2f}" if results['served'] > 0 else "NA")
        c3.metric("90th percentile time (min)", f"{results['p90_time_min']:.2f}" if results['served'] >= 10 else "NA")

        # Display utilization bar chart
        util_data = pd.DataFrame({
            'Resource': [
                'Kiosk', 'Register', 'Cook', 'Expo', 'Drink', 'Condiment', 'Tables'
            ],
            'Utilization': [
                results['util_kiosks'],
                results['util_registers'],
                results['util_cooks'],
                results['util_expo'],
                results['util_drinks'],
                results['util_condiments'],
                results['util_tables'],
            ],
        })
        st.subheader("Resource Utilization")
        st.bar_chart(util_data.set_index('Resource'))

        # Display average queue lengths bar chart
        qlen_data = pd.DataFrame({
            'Resource': [
                'Kiosk', 'Register', 'Cook', 'Expo', 'Drink', 'Condiment', 'Tables'
            ],
            'Avg Queue Length': [
                results['qlen_kiosks'],
                results['qlen_registers'],
                results['qlen_cooks'],
                results['qlen_expo'],
                results['qlen_drinks'],
                results['qlen_condiments'],
                results['qlen_tables'],
            ],
        })
        st.subheader("Average Queue Lengths")
        st.bar_chart(qlen_data.set_index('Resource'))

        # Display the animated GIF
        st.subheader(f"Animation (first {anim_minutes} minutes)")
        st.image(gif_bytes, caption="Customer movement and queue lengths", use_column_width=True)

        st.write(
            "The simulation model is stochastic; running it multiple times may\n"
            "yield slightly different results. Adjust the parameters and click\n"
            "**Run Simulation and Animate** again to explore different scenarios."
        )


if __name__ == "__main__":
    main()