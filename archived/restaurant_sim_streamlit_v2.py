"""
Restaurant Simulation with Interactive p5.js Animation for Streamlit
===================================================================

This Streamlit app simulates the operations of a fast‑casual restaurant
using a discrete event simulation (DES) implemented in pure Python and
salabim. In addition to producing the familiar numerical summary
metrics and bar charts for resource utilisation and queue lengths, it
also renders a smooth, 2D, birds‑eye‑view animation of customers
walking through the restaurant.

The animation is powered by `p5.js`, a JavaScript framework for
creative coding. The simulation runs entirely in Python and produces
frame data (customer positions, queue lengths, and busy server counts)
on a regular time grid. This data is passed to a custom HTML canvas
embedded in the Streamlit interface via `st.components.v1.html`. The
JavaScript code reads the frame data and draws moving circles for
customers, vertical queue bars at each station, and busy counters at
the top of the screen.

To run this app, install the dependencies listed in ``requirements.txt``
and execute:

    streamlit run restaurant_sim_streamlit_p5.py

Adjust the simulation parameters in the sidebar, click **Run Simulation
and Animate**, and watch customers move through the restaurant while
queue lengths and busy fractions update in real time. Because the
animation runs entirely client‑side in the browser, it is responsive
and can scale to many customers without overloading the server.
"""

import json
import random
import statistics
from typing import Dict, List, Tuple

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
        # Customer does not wait; queue length does not change (record zero change)
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
    # Define node coordinates based on the earlier animation positions
    # and normalise them to [0, 10] x [0, 4] for plotting convenience.
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


def generate_animation_frames(
    segments: List[List[Dict[str, float]]],
    queue_ts: Dict[str, np.ndarray],
    busy_ts: Dict[str, np.ndarray],
    capacities: Dict[str, int],
    anim_seconds: float,
    dt: float = 0.5,
) -> Tuple[List[List[Tuple[float, float]]], Dict[str, List[float]], Dict[str, List[float]]]:
    """Generate per-frame positions and queue/busy series at a specified time step.

    Parameters
    ----------
    segments : List[List[Dict]]
        Customer movement and wait segments from ``simulate_customers``.
    queue_ts : Dict[str, np.ndarray]
        Queue lengths sampled every second.
    busy_ts : Dict[str, np.ndarray]
        Busy server counts sampled every second.
    capacities : Dict[str, int]
        Capacity of each resource (used to compute busy fractions).
    anim_seconds : float
        Duration of the animation in seconds.
    dt : float, optional
        Time step (seconds) between frames.  Default is 0.5s for smooth animation.

    Returns
    -------
    frames : List[List[Tuple[float, float]]]
        A list of frames; each frame is a list of (x, y) positions for customers
        active during that time step.  Coordinates are on a normalised scale
        [0, 10] x [0, 4].  Customers who have finished their path before a
        given frame are excluded from subsequent frames.
    queue_series : Dict[str, List[float]]
        Queue lengths for each resource at each frame (same length as frames).
    busy_series : Dict[str, List[float]]
        Busy server counts for each resource at each frame (same length as frames).
    """
    num_frames = int(anim_seconds / dt) + 1
    times = np.linspace(0.0, anim_seconds, num_frames)

    # Prepare per-customer state to track which segment index applies at time t
    customer_states = []
    for cust_segments in segments:
        if cust_segments:
            customer_states.append({'segments': cust_segments, 'index': 0})

    frames: List[List[Tuple[float, float]]] = []
    # Queue and busy series lists
    queue_series: Dict[str, List[float]] = {key: [] for key in queue_ts.keys()}
    busy_series: Dict[str, List[float]] = {key: [] for key in busy_ts.keys()}

    for t in times:
        positions: List[Tuple[float, float]] = []
        for state in customer_states:
            segs = state['segments']
            i = state['index']
            # Advance segment index if t >= end_time
            while i < len(segs) and t >= segs[i]['end_time']:
                i += 1
            state['index'] = i
            if i < len(segs) and segs[i]['start_time'] <= t < segs[i]['end_time']:
                seg = segs[i]
                if seg['wait'] or seg['end_time'] == seg['start_time']:
                    positions.append((seg['x0'], seg['y0']))
                else:
                    # Linear interpolation along the path
                    frac = (t - seg['start_time']) / (seg['end_time'] - seg['start_time']) if seg['end_time'] > seg['start_time'] else 0.0
                    x = seg['x0'] + frac * (seg['x1'] - seg['x0'])
                    y = seg['y0'] + frac * (seg['y1'] - seg['y0'])
                    positions.append((x, y))
        frames.append(positions)
        # Append queue lengths (sampled at integer seconds)
        idx_sec = int(min(max(int(t), 0), len(queue_ts['kiosks']) - 1))
        for key in queue_ts.keys():
            queue_series[key].append(float(queue_ts[key][idx_sec]))
        # Append busy server counts (sampled at integer seconds)
        for key in busy_ts.keys():
            busy_series[key].append(float(busy_ts[key][idx_sec]))

    return frames, queue_series, busy_series


# ----------------------------------------------------------------------------
# Streamlit interface
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Restaurant DES with p5.js Animation", layout="wide")
    st.title("🍽️ Restaurant Discrete Event Simulation with Interactive Animation")
    st.write(
        "This interactive app simulates a fast casual restaurant. Adjust the\n"
        "parameters in the sidebar and click **Run Simulation and Animate**\n"
        "to see how changes affect throughput, waiting times, and resource utilisation.\n"
        "An interactive animation shows customers walking through the restaurant\n"
        "layout, queues forming at each station, and real‑time busy fractions for\n"
        "registers, cooks and expo staff. The animation is rendered client‑side\n"
        "using p5.js for smooth, responsive visuals."
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
        "Total seats (sum over tables)", min_value=0, max_value=200,
        value=sum({2: 18, 4: 10}.values()) * 2, step=2
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
            # Generate frame data for the animation (0.5s time step for smoothness)
            dt = 0.5
            frames, queue_series, busy_series = generate_animation_frames(
                segments=segs,
                queue_ts=qts,
                busy_ts=bts,
                capacities=caps,
                anim_seconds=anim_seconds,
                dt=dt,
            )
            # Convert frames to a serialisable structure: list of lists of [x, y]
            frames_list = [[list(pos) for pos in frame] for frame in frames]
            # Convert queue/busy series to lists
            queue_json = {k: [float(x) for x in v] for k, v in queue_series.items()}
            busy_json = {k: [float(x) for x in v] for k, v in busy_series.items()}

            # Node positions for JavaScript (normalised [0,10]x[0,4])
            # These must correspond to the positions used in simulate_customers
            node_coords = {
                'door':      (50, 250),
                'kiosk':     (250, 120),
                'register':  (250, 380),
                'pickup':    (480, 250),
                'drink':     (680, 200),
                'condiment': (820, 220),
                'seating':   (1050, 260),
                'exit':      (1250, 260),
            }
            xs = [p[0] for p in node_coords.values()]
            ys = [p[1] for p in node_coords.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            def norm_js(x: float, y: float) -> Tuple[float, float]:
                return ((x - min_x) / (max_x - min_x) * 10.0,
                        (y - min_y) / (max_y - min_y) * 4.0)
            node_js = {name: norm_js(*pos) for name, pos in node_coords.items()}

            # Prepare capacities for busy counters
            busy_caps = {
                'registers': caps.get('registers', 1),
                'cooks': caps.get('cooks', 1),
                'expo': caps.get('expo', 1),
            }

            # Assemble HTML and JavaScript for animation
            # Encode data as JSON
            frames_json = json.dumps(frames_list)
            queue_json_str = json.dumps(queue_json)
            busy_json_str = json.dumps(busy_json)
            node_json_str = json.dumps(node_js)
            busy_caps_str = json.dumps(busy_caps)

            # Determine canvas size (scaled from normalised coordinates)
            canvas_width = 800
            canvas_height = 320
            # JavaScript code for p5.js animation.  Use instance mode to avoid global pollution.
            # Build the JavaScript and HTML code using str.format to avoid
            # accidental f-string interpolation of braces.  We double the
            # braces in the JavaScript templates to escape them for .format().
            js_template = """
            <script src="https://cdn.jsdelivr.net/npm/p5@1.4.2/lib/p5.min.js"></script>
            <div id="p5-container"></div>
            <script>
            const frames = {frames_json};
            const queueSeries = {queue_json};
            const busySeries = {busy_json};
            const nodePos = {node_json};
            const busyCaps = {busy_caps};
            const dt = {dt_value};
            const fps = {fps_value};
            const canvasW = {canvas_w};
            const canvasH = {canvas_h};

            // Create the p5 sketch
            const sketch = (p) => {{
              let frameIndex = 0;
              p.setup = () => {{
                p.createCanvas(canvasW, canvasH);
                p.frameRate(fps);
              }};
              p.draw = () => {{
                p.background(255);
                // Draw static nodes (scaled from normalised positions)
                const rectW = 60;
                const rectH = 40;
                const scaleX = canvasW / 10.0;
                const scaleY = canvasH / 4.0;
                // Node labels and positions
                const labels = {{
                  'door': 'Door',
                  'kiosk': 'Kiosk',
                  'register': 'Register',
                  'pickup': 'Pickup',
                  'drink': 'Drink',
                  'condiment': 'Condiment',
                  'seating': 'Seating',
                  'exit': 'Exit'
                }};
                for (const [name, pos] of Object.entries(nodePos)) {{
                  const x = pos[0] * scaleX;
                  const y = pos[1] * scaleY;
                  p.fill(230);
                  p.stroke(180);
                  p.rect(x - rectW/2, y - rectH/2, rectW, rectH, 5);
                  p.fill(50);
                  p.noStroke();
                  p.textAlign(p.CENTER, p.CENTER);
                  p.textSize(12);
                  p.text(labels[name], x, y);
                }}
                // Draw queue bars and counts
                const barScale = 10; // pixels per person in queue
                const barWidth = 8;
                // Specific bar positions relative to stations
                const barPositions = {{
                  'kiosks': nodePos['kiosk'],
                  'registers': nodePos['register'],
                  'cooks': [nodePos['pickup'][0] - 0.3, nodePos['pickup'][1]],
                  'expo': [nodePos['pickup'][0] + 0.3, nodePos['pickup'][1]],
                  'drinks': nodePos['drink'],
                  'condiments': nodePos['condiment'],
                  'tables': nodePos['seating']
                }};
                for (const [key, pos] of Object.entries(barPositions)) {{
                  const q = queueSeries[key][frameIndex] || 0;
                  const x = pos[0] * scaleX;
                  const y = pos[1] * scaleY;
                  const h = q * barScale;
                  p.fill(120);
                  p.rect(x - barWidth/2, y - rectH/2 - h - 5, barWidth, h);
                  p.fill(50);
                  p.textSize(8);
                  p.textAlign(p.CENTER, p.BOTTOM);
                  p.text('Q: ' + Math.floor(q), x, y - rectH/2 - h - 8);
                }}
                // Draw busy counters at top
                const br = busySeries['registers'][frameIndex] || 0;
                const bc = busySeries['cooks'][frameIndex] || 0;
                const be = busySeries['expo'][frameIndex] || 0;
                const txt = 'Registers: ' + Math.floor(br) + '/' + busyCaps['registers'] + '   Cooks: ' + Math.floor(bc) + '/' + busyCaps['cooks'] + '   Expo: ' + Math.floor(be) + '/' + busyCaps['expo'];
                p.fill(50);
                p.textSize(12);
                p.textAlign(p.LEFT, p.TOP);
                p.text(txt, 10, 10);
                // Draw customers
                const positions = frames[frameIndex] || [];
                p.fill(0, 102, 204);
                p.noStroke();
                for (const pos of positions) {{
                  const x = pos[0] * scaleX;
                  const y = pos[1] * scaleY;
                  p.ellipse(x, y, 10, 10);
                }}
                // Advance frame
                frameIndex++;
                if (frameIndex >= frames.length) {{
                  frameIndex = frames.length - 1; // stop at last frame
                }}
              }};
            }};
            // Create a new p5 instance
            new p5(sketch, document.getElementById('p5-container'));
            </script>
            """
            js_code = js_template.format(
                frames_json=frames_json,
                queue_json=queue_json_str,
                busy_json=busy_json_str,
                node_json=node_json_str,
                busy_caps=busy_caps_str,
                dt_value=dt,
                fps_value=fps,
                canvas_w=canvas_width,
                canvas_h=canvas_height,
            )
            # Render the dynamic content
            st.subheader("Summary Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Customers served", results["served"])
            c2.metric("Avg time in system (min)", f"{results['avg_time_min']:.2f}" if results['served'] > 0 else "NA")
            c3.metric("90th percentile time (min)", f"{results['p90_time_min']:.2f}" if results['served'] >= 10 else "NA")

            # Display utilisation bar chart
            util_data = pd.DataFrame({
                'Resource': [
                    'Kiosk', 'Register', 'Cook', 'Expo', 'Drink', 'Condiment', 'Tables'
                ],
                'Utilisation': [
                    results['util_kiosks'],
                    results['util_registers'],
                    results['util_cooks'],
                    results['util_expo'],
                    results['util_drinks'],
                    results['util_condiments'],
                    results['util_tables'],
                ],
            })
            st.subheader("Resource Utilisation")
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

            # Display the animation
            st.subheader(f"Animation (first {anim_minutes} minutes)")
            # Use components.html to embed the p5.js animation
            st.components.v1.html(js_code, height=canvas_height + 20, width=canvas_width)


if __name__ == "__main__":
    main()