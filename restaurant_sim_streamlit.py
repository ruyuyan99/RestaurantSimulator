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


def schedule_service(res_state: ResourceState, arrival_time: float, mean_service: float) -> Tuple[int, float, float]:
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
    (server_index, start_service_time, end_service_time)
        server_index : int
            Index of the server (0‑based) assigned to this customer.
        start_service_time : float
            Time when service begins.
        end_service_time : float
            Time when service finishes.
    """
    if res_state.capacity <= 0:
        # No capacity at this resource; service happens immediately with zero duration
        return (0, arrival_time, arrival_time)

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
    return (idx, start_service, end_service)


def simulate_customers(
    sim_hours: float,
    arrival_rate: float,
    capacities: Dict[str, int],
    pct_to_kiosk: float,
    animation_seconds: float,
    station_coords: Dict[str, List[Tuple[float, float]]] = None,
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
    # Normalise base positions into a rectangular canvas with generous margins.
    # We compress the x‑range into [0.8, 9.2] (instead of [0.5, 9.5]) and the y‑range
    # into [0.5, 3.5].  This leaves extra horizontal margin so station
    # rectangles (50 px wide) do not get cut off at the edges when the
    # animation is drawn.  These values mirror those used in the JS
    # normalisation helper.
    xs = [p[0] for p in raw_pos.values()]
    ys = [p[1] for p in raw_pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    def norm(x: float, y: float) -> Tuple[float, float]:
        # compress to 8.4 units (9.2-0.8) horizontally with 0.8 margin on left and right
        x_norm = (x - min_x) / (max_x - min_x) * 8.4 + 0.8
        # compress y into [0.5,3.5]
        y_norm = (y - min_y) / (max_y - min_y) * 3.0 + 0.5
        return (x_norm, y_norm)
    NODE_COORDS = {node: norm(*pos) for node, pos in raw_pos.items()}

    # If station_coords is not provided, we derive per‑server positions
    # using the base NODE_COORDS for each resource group.  When provided,
    # station_coords overrides NODE_COORDS for assigning positions during
    # waiting and service segments.  station_coords maps resource keys
    # (e.g. 'kiosks') to a list of (x, y) tuples, one per server.  These
    # coordinates should be normalised to the same scale as NODE_COORDS.
    if station_coords is None:
        station_coords = {}
        import math
        def layout_positions(base: Tuple[float, float], n: int, max_cols: int = 3, spacing_x: float = 0.4, spacing_y: float = 0.4) -> List[Tuple[float, float]]:
            if n <= 0:
                return []
            cols = min(n, max_cols)
            rows = int(math.ceil(n / cols))
            positions = []
            for i in range(n):
                row = i // cols
                col = i % cols
                dx = (col - (cols - 1) / 2.0) * spacing_x
                dy = (row - (rows - 1) / 2.0) * spacing_y
                positions.append((base[0] + dx, base[1] + dy))
            return positions
        # Map resource keys to their base node keys for deriving base positions
        resource_base = {
            'kiosks':     KIOSK,
            'registers':  REGISTER,
            'cooks':      PICKUP,
            'expo':       PICKUP,
            'drinks':     DRINK,
            'condiments': CONDIMENT,
            'tables':     SEATING,
        }
        for key, base_node in resource_base.items():
            count = capacities.get(key, 0)
            if count <= 0:
                station_coords[key] = []
            else:
                base_pos = NODE_COORDS[base_node]
                # Determine station positions.  For expo, enforce a single
                # horizontal row regardless of server count.  For cooks with
                # expo servers, split cooks into a top band (first three) and
                # one or more bottom bands, separated from the expo band by a
                # fixed vertical offset.  Otherwise, fall back to a regular
                # grid layout.
                if key == 'expo':
                    # Expo stations are always laid out on a single row.  We
                    # compute horizontal offsets based on the number of expo
                    # servers so they are evenly spaced about the base
                    # position.  All expo stations share the same y‑coordinate.
                    positions: List[Tuple[float, float]] = []
                    for i in range(count):
                        dx = (i - (count - 1) / 2.0) * 0.4
                        positions.append((base_pos[0] + dx, base_pos[1]))
                    station_coords[key] = positions
                elif key == 'cooks' and capacities.get('expo', 0) > 0:
                    # When expo servers exist, cooks must not occupy the same
                    # horizontal band as the expo.  We use a grid of up to
                    # three columns for cooks and assign the first three
                    # cooks to a band above the expo and the remaining cooks
                    # to bands below the expo.  Rows below the expo are
                    # spaced at multiples of 0.4 (the vertical spacing
                    # used in layout_positions) relative to the expo band.
                    tmp_positions = layout_positions(base_pos, count, max_cols=3)
                    positions: List[Tuple[float, float]] = []
                    for i, (cx, cy_unused) in enumerate(tmp_positions):
                        if i < 3:
                            # Top band: shift cooks up by 0.4
                            positions.append((cx, base_pos[1] - 0.4))
                        else:
                            # Bottom bands: shift cooks down by (row_index+1)*0.4
                            idx_bottom = i - 3
                            row_index = idx_bottom // 3
                            positions.append((cx, base_pos[1] + (row_index + 1) * 0.4))
                    station_coords[key] = positions
                else:
                    # Default layout for other resources or when no expo exists
                    station_coords[key] = layout_positions(base_pos, count)

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
            svc_mean = MEAN_ORDER_TIME
            server_idx, start_svc, end_svc = schedule_service(resources['kiosks'], current_time, svc_mean)
            # Wait segment if the customer arrives before service can start
            if start_svc > current_time:
                # Use per‑station coordinates if available, otherwise fall back to base
                if station_coords.get('kiosks'):
                    px, py = station_coords['kiosks'][server_idx]
                else:
                    px, py = NODE_COORDS[current_node]
                customer_segments.append({
                    'start_time': current_time,
                    'end_time': start_svc,
                    'x0': px,
                    'y0': py,
                    'x1': px,
                    'y1': py,
                    'wait': True,
                })
            # Service segment
            if end_svc > start_svc:
                if station_coords.get('kiosks'):
                    px, py = station_coords['kiosks'][server_idx]
                else:
                    px, py = NODE_COORDS[current_node]
                customer_segments.append({
                    'start_time': start_svc,
                    'end_time': end_svc,
                    'x0': px,
                    'y0': py,
                    'x1': px,
                    'y1': py,
                    'wait': True,
                })
            current_time = end_svc
        else:
            svc_mean = MEAN_REGISTER_TIME
            server_idx, start_svc, end_svc = schedule_service(resources['registers'], current_time, svc_mean)
            if start_svc > current_time:
                if station_coords.get('registers'):
                    px, py = station_coords['registers'][server_idx]
                else:
                    px, py = NODE_COORDS[current_node]
                customer_segments.append({
                    'start_time': current_time,
                    'end_time': start_svc,
                    'x0': px,
                    'y0': py,
                    'x1': px,
                    'y1': py,
                    'wait': True,
                })
            if end_svc > start_svc:
                if station_coords.get('registers'):
                    px, py = station_coords['registers'][server_idx]
                else:
                    px, py = NODE_COORDS[current_node]
                customer_segments.append({
                    'start_time': start_svc,
                    'end_time': end_svc,
                    'x0': px,
                    'y0': py,
                    'x1': px,
                    'y1': py,
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

        # Service at cook (use per‑server positions if defined)
        srv_idx, start_svc, end_svc = schedule_service(resources['cooks'], current_time, MEAN_COOK_TIME)
        # Wait segment before cook
        if start_svc > current_time:
            if station_coords.get('cooks'):
                px, py = station_coords['cooks'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
                'wait': True,
            })
        # Service segment at cook
        if end_svc > start_svc:
            if station_coords.get('cooks'):
                px, py = station_coords['cooks'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
                'wait': True,
            })
        current_time = end_svc

        # Service at expo
        srv_idx, start_svc, end_svc = schedule_service(resources['expo'], current_time, MEAN_EXPO_TIME)
        if start_svc > current_time:
            if station_coords.get('expo'):
                px, py = station_coords['expo'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
                'wait': True,
            })
        if end_svc > start_svc:
            if station_coords.get('expo'):
                px, py = station_coords['expo'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
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
        srv_idx, start_svc, end_svc = schedule_service(resources['drinks'], current_time, MEAN_DRINK_TIME)
        if start_svc > current_time:
            if station_coords.get('drinks'):
                px, py = station_coords['drinks'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
                'wait': True,
            })
        if end_svc > start_svc:
            if station_coords.get('drinks'):
                px, py = station_coords['drinks'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
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
        srv_idx, start_svc, end_svc = schedule_service(resources['condiments'], current_time, MEAN_CONDIMENT_TIME)
        if start_svc > current_time:
            if station_coords.get('condiments'):
                px, py = station_coords['condiments'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
                'wait': True,
            })
        if end_svc > start_svc:
            if station_coords.get('condiments'):
                px, py = station_coords['condiments'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
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
        srv_idx, start_svc, end_svc = schedule_service(resources['tables'], current_time, dine_t)
        # Wait segment before dining
        if start_svc > current_time:
            if station_coords.get('tables'):
                px, py = station_coords['tables'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': current_time,
                'end_time': start_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
                'wait': True,
            })
        # Dining segment
        if end_svc > start_svc:
            if station_coords.get('tables'):
                px, py = station_coords['tables'][srv_idx]
            else:
                px, py = NODE_COORDS[current_node]
            customer_segments.append({
                'start_time': start_svc,
                'end_time': end_svc,
                'x0': px,
                'y0': py,
                'x1': px,
                'y1': py,
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
    """
    Entry point for the Streamlit application.

    The layout is organised into two tabs: **Animation** and **Summary Results**.
    Users configure parameters in the sidebar and click the run button to
    execute the simulation.  The animation tab displays a p5.js canvas
    showing the restaurant layout and animated customers, along with
    pause/resume and skip controls.  A static skip button allows
    users to bypass the animation entirely.  The summary tab presents
    throughput and performance metrics after the simulation completes.
    """
    # Declare global variables up front so that assignments within this
    # function modify the module‑level constants. The global statements
    # must precede any use of these names within the function
    # otherwise Python will treat them as local variables and raise
    # a SyntaxError if they are referenced prior to the global declaration.
    global WALK_SPEED_MPS, MEAN_ORDER_TIME, MEAN_REGISTER_TIME, MEAN_COOK_TIME
    global MEAN_EXPO_TIME, MEAN_DRINK_TIME, MEAN_CONDIMENT_TIME, MEAN_DINE_TIME_MIN
    global PARTY_SIZE_WEIGHTS

    st.set_page_config(page_title="Restaurant DES with p5.js Animation", layout="wide")
    st.title("🍽️ Restaurant Discrete Event Simulation with Interactive Animation")
    st.write(
        "This interactive app simulates a fast casual restaurant. Adjust the\n"
        "parameters in the sidebar and click **Run Simulation and Animate**\n"
        "to see how changes affect throughput, waiting times, and resource utilisation.\n"
        "A 2D animation shows customers walking through the restaurant layout,\n"
        "queues forming at each station, and real‑time busy fractions for\n"
        "registers, cooks and expo staff."
    )

    # Sidebar controls
    # Organise sidebar controls into expandable sections.  This avoids an overly long
    # sidebar while still exposing all simulation parameters.
    with st.sidebar.expander("Simulation parameters", expanded=True):
        sim_hours = st.slider("Simulation duration (hours)", 1, 12, DEFAULT_SIM_HOURS)
        arrival_rate = st.slider(
            "Arrival rate (customers per minute)", 0.1, 5.0, DEFAULT_ARRIVAL_RATE, 0.1
        )
        pct_to_kiosk = st.slider(
            "Fraction choosing kiosk", 0.0, 1.0, 0.75, 0.05
        )
        walking_speed = st.number_input(
            "Walking speed (m/s)", min_value=0.1, max_value=5.0, value=WALK_SPEED_MPS, step=0.1,
            help="Average walking speed of customers."
        )
        st.markdown("**Party size distribution (weights should sum to 1)**")
        # Display party size weights on separate rows for improved readability.
        weight1 = st.number_input(
            "Party of 1", min_value=0.0, max_value=1.0,
            value=PARTY_SIZE_WEIGHTS[0][1], step=0.05,
            key="ps_weight1"
        )
        weight2 = st.number_input(
            "Party of 2", min_value=0.0, max_value=1.0,
            value=PARTY_SIZE_WEIGHTS[1][1], step=0.05,
            key="ps_weight2"
        )
        weight3 = st.number_input(
            "Party of 3", min_value=0.0, max_value=1.0,
            value=PARTY_SIZE_WEIGHTS[2][1], step=0.05,
            key="ps_weight3"
        )
        weight4 = st.number_input(
            "Party of 4", min_value=0.0, max_value=1.0,
            value=PARTY_SIZE_WEIGHTS[3][1], step=0.05,
            key="ps_weight4"
        )
    with st.sidebar.expander("Resource capacities", expanded=True):
        n_kiosks = st.number_input(
            "Number of kiosks", min_value=0, max_value=20, value=6, step=1
        )
        n_registers = st.number_input(
            "Number of registers", min_value=0, max_value=20, value=2, step=1
        )
        n_cooks = st.number_input(
            "Number of cooks", min_value=0, max_value=20, value=5, step=1
        )
        n_expo = st.number_input(
            "Number of expo staff", min_value=0, max_value=20, value=1, step=1
        )
        n_drinks = st.number_input(
            "Number of drink stations", min_value=0, max_value=20, value=2, step=1
        )
        n_condiments = st.number_input(
            "Number of condiment stations", min_value=0, max_value=20, value=2, step=1
        )
        table_cap = st.number_input(
            "Total seats (sum over tables)", min_value=0, max_value=200,
            value=sum({2: 18, 4: 10}.values()) * 2, step=2
        )
    with st.sidebar.expander("Service times (mean in seconds)", expanded=True):
        mean_order_time = st.number_input("Order time at kiosk", min_value=1, max_value=600, value=MEAN_ORDER_TIME, step=1)
        mean_register_time = st.number_input("Order time at register", min_value=1, max_value=600, value=MEAN_REGISTER_TIME, step=1)
        mean_cook_time = st.number_input("Cook time", min_value=1, max_value=600, value=MEAN_COOK_TIME, step=1)
        mean_expo_time = st.number_input("Expo time", min_value=1, max_value=600, value=MEAN_EXPO_TIME, step=1)
        mean_drink_time = st.number_input("Drink station time", min_value=1, max_value=600, value=MEAN_DRINK_TIME, step=1)
        mean_condiment_time = st.number_input("Condiment station time", min_value=1, max_value=600, value=MEAN_CONDIMENT_TIME, step=1)
        st.markdown("**Dining times (mean minutes) by party size**")
        dine1 = st.number_input("Party of 1", min_value=1.0, max_value=120.0, value=float(MEAN_DINE_TIME_MIN.get(1, 14)), step=1.0)
        dine2 = st.number_input("Party of 2", min_value=1.0, max_value=120.0, value=float(MEAN_DINE_TIME_MIN.get(2, 20)), step=1.0)
        dine3 = st.number_input("Party of 3", min_value=1.0, max_value=120.0, value=float(MEAN_DINE_TIME_MIN.get(3, 24)), step=1.0)
        dine4 = st.number_input("Party of 4", min_value=1.0, max_value=120.0, value=float(MEAN_DINE_TIME_MIN.get(4, 28)), step=1.0)
    with st.sidebar.expander("Animation controls", expanded=True):
        skip_animation = st.checkbox(
            "Skip animation",
            value=False,
            help="Check to bypass the animation and display only summary results."
        )
        fps = st.select_slider(
            "Animation FPS", options=[5, 10, 15], value=10
        )
        sim_speedup = st.slider(
            "Simulation speedup (minutes per second)", min_value=1, max_value=10, value=1, step=1,
            help="Number of minutes of simulated time compressed into one second of animation."
        )
    # Trigger to run simulation.  When pressed, results will be stored in session_state
    run_button = st.sidebar.button("Run Simulation and Animate")

    # Use Streamlit tabs to separate animation from summary results
    tab_animation, tab_summary = st.tabs(["Animation", "Summary Results"])

    # When run_button is clicked, run the simulations and store results in session_state
    if run_button:
        with st.spinner("Running simulation and preparing data, please wait..."):
            # ------------------------------------------------------------------
            # Update global simulation parameters based on sidebar inputs.  These
            # variables control walking speed, party size distribution and
            # service times.  We normalise party size weights so that they
            # sum to 1.  Note that the global declarations for these variables
            # are at the top of ``main``.
            WALK_SPEED_MPS = float(walking_speed)
            MEAN_ORDER_TIME = float(mean_order_time)
            MEAN_REGISTER_TIME = float(mean_register_time)
            MEAN_COOK_TIME = float(mean_cook_time)
            MEAN_EXPO_TIME = float(mean_expo_time)
            MEAN_DRINK_TIME = float(mean_drink_time)
            MEAN_CONDIMENT_TIME = float(mean_condiment_time)
            # Update dining times dictionary
            MEAN_DINE_TIME_MIN = {
                1: float(dine1),
                2: float(dine2),
                3: float(dine3),
                4: float(dine4),
            }
            # Normalise party size weights; if the sum is zero, keep defaults
            total_weight = weight1 + weight2 + weight3 + weight4
            if total_weight > 0:
                PARTY_SIZE_WEIGHTS = [
                    (1, float(weight1) / total_weight),
                    (2, float(weight2) / total_weight),
                    (3, float(weight3) / total_weight),
                    (4, float(weight4) / total_weight),
                ]
            # Run simulation using salabim for summary metrics
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
            anim_seconds = sim_hours * 3600.0
            # Run custom simulation to obtain segments and queue/busy time series
            segs, qts, bts = simulate_customers(
                sim_hours=sim_hours,
                arrival_rate=arrival_rate,
                capacities=caps,
                pct_to_kiosk=pct_to_kiosk,
                animation_seconds=anim_seconds,
            )
            # Determine dt based on speedup and FPS
            dt = (sim_speedup * 60.0) / float(fps)
            frames, queue_series, busy_series = generate_animation_frames(
                segments=segs,
                queue_ts=qts,
                busy_ts=bts,
                capacities=caps,
                anim_seconds=anim_seconds,
                dt=dt,
            )
            # Serialise frame and series data for JavaScript
            frames_list = [[list(pos) for pos in frame] for frame in frames]
            queue_json = {k: [float(x) for x in v] for k, v in queue_series.items()}
            busy_json = {k: [float(x) for x in v] for k, v in busy_series.items()}

            # ------------------------------------------------------------------
            # Compute dynamic node positions for each station group.  We normalise
            # the original node positions to [0,10]x[0,4] and then create grid
            # layouts for resources with multiple stations (e.g. kiosks, registers,
            # cooks, expo, drinks, condiments, tables).  Each station is given a
            # unique key like 'kiosks_0', 'kiosks_1', etc.  Additional metadata
            # such as pastel colours and labels are also prepared and passed to
            # the JavaScript renderer.

            # Normalised base positions (same scaling used in simulate_customers)
            base_coords = {
                'door':      (50, 250),
                'kiosk':     (250, 120),
                'register':  (250, 380),
                'pickup':    (480, 250),
                'drink':     (680, 200),
                'condiment': (820, 220),
                'seating':   (1050, 260),
                'exit':      (1250, 260),
            }
            xs = [p[0] for p in base_coords.values()]
            ys = [p[1] for p in base_coords.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            # Normalisation helper for JavaScript node positions.  To avoid
            # stations being cut off at the extreme left and right edges of
            # the canvas, we compress the x‑range from [0, 10] to [0.5, 9.5].
            # Similarly, compress the y‑range slightly so nodes are not flush
            # against the top or bottom.  This adds a margin around the
            # layout, ensuring that labels and buttons do not get cut off.
            def norm_js(x: float, y: float) -> Tuple[float, float]:
                """
                Normalise raw node coordinates into the animation coordinate
                system.  To ensure stations are not cut off at the edges, we
                compress the x‑range into [0.8, 9.2] and the y‑range into
                [0.5, 3.5].  These ranges match those used in the Python
                simulation for station positions.  The extra horizontal
                margin prevents station rectangles and labels from being
                clipped when drawn in the canvas.
                """
                x_norm = (x - min_x) / (max_x - min_x) * 8.4 + 0.8
                y_norm = (y - min_y) / (max_y - min_y) * 3.0 + 0.5
                return (x_norm, y_norm)
            base_norm = {name: norm_js(*pos) for name, pos in base_coords.items()}

            import math
            # Helper to lay out stations in a grid around a base position
            def layout_positions(base: Tuple[float, float], n: int, max_cols: int = 3, spacing_x: float = 0.4, spacing_y: float = 0.4) -> List[Tuple[float, float]]:
                if n <= 0:
                    return []
                cols = min(n, max_cols)
                rows = int(math.ceil(n / cols))
                positions = []
                for i in range(n):
                    row = i // cols
                    col = i % cols
                    dx = (col - (cols - 1) / 2.0) * spacing_x
                    dy = (row - (rows - 1) / 2.0) * spacing_y
                    positions.append((base[0] + dx, base[1] + dy))
                return positions

            # Map resource keys to their corresponding base node keys
            resource_base = {
                'kiosks': 'kiosk',
                'registers': 'register',
                'cooks': 'pickup',
                'expo': 'pickup',
                'drinks': 'drink',
                'condiments': 'condiment',
                'tables': 'seating',
            }
            # Pastel colour palette for each resource group (hex strings)
            group_colours = {
                'kiosks': '#cfe2f3',    # light blue
                'registers': '#f4cccc', # light red
                'cooks': '#fff2cc',     # light yellow
                'expo': '#d9ead3',      # light green
                'drinks': '#ead1dc',    # light pink
                'condiments': '#d0e0e3',# light cyan
                'tables': '#d9d2e9',    # light purple
                'door': '#eeeeee',
                'exit': '#eeeeee',
            }
            # Compute node positions and labels
            node_positions: Dict[str, Tuple[float, float]] = {}
            node_labels: Dict[str, str] = {}
            node_colors: Dict[str, str] = {}
            # Door and Exit have single positions
            node_positions['door'] = base_norm['door']
            node_labels['door'] = 'Door'
            node_colors['door'] = group_colours['door']
            node_positions['exit'] = base_norm['exit']
            node_labels['exit'] = 'Exit'
            node_colors['exit'] = group_colours['exit']
            # Compute positions for each resource group with capacity > 0
            for res_key, base_key in resource_base.items():
                count = caps.get(res_key, 0)
                if count <= 0:
                    continue
                base_pos = base_norm[base_key]
                # For expo, enforce a single horizontal band regardless of
                # server count.  For cooks with expo servers present, split
                # cooks into a top band (first three) and one or more
                # bottom bands separated from the expo band.  Otherwise,
                # positions are laid out in a regular grid.
                positions: List[Tuple[float, float]]
                if res_key == 'expo':
                    positions = []
                    for i in range(count):
                        dx = (i - (count - 1) / 2.0) * 0.4
                        positions.append((base_pos[0] + dx, base_pos[1]))
                elif res_key == 'cooks' and caps.get('expo', 0) > 0:
                    # Generate preliminary x positions using up to three
                    # columns.  We ignore the y coordinate returned by
                    # layout_positions and instead compute our own y offsets.
                    tmp_positions = layout_positions(base_pos, count, max_cols=3)
                    positions = []
                    for i, (px_tmp, _py_tmp) in enumerate(tmp_positions):
                        if i < 3:
                            # Top band above expo
                            positions.append((px_tmp, base_pos[1] - 0.4))
                        else:
                            idx_bottom = i - 3
                            row_index = idx_bottom // 3
                            positions.append((px_tmp, base_pos[1] + (row_index + 1) * 0.4))
                else:
                    positions = layout_positions(base_pos, count)
                for idx, (px, py) in enumerate(positions):
                    node_key = f"{res_key}_{idx}"
                    node_positions[node_key] = (px, py)
                    # Create label with capitalised name and number
                    label_name = res_key[:-1].capitalize() if res_key.endswith('s') else res_key.capitalize()
                    node_labels[node_key] = f"{label_name} {idx+1}"
                    node_colors[node_key] = group_colours.get(res_key, '#cccccc')

            # No additional cook offset is applied here.  Cook stations are
            # repositioned within the loop above when expo servers exist.  This
            # ensures that cooks are placed on rows above or below the expo
            # band without further adjustment.

            # Busy counters show only registers, cooks and expo
            busy_caps = {
                'registers': caps.get('registers', 1),
                'cooks': caps.get('cooks', 1),
                'expo': caps.get('expo', 1),
            }
            # ------------------------------------------------------------------
            # Load PNG icons and prepare data URIs for each resource type.  We
            # embed these as base64 strings so they can be loaded by p5.js in
            # the browser without separate file requests.  The icons reside in
            # the ``assets`` folder relative to this script (JPEG versions have
            # been moved to an ``assets/jpegs`` subfolder and are ignored).  We map
            # plural resource keys to singular filenames for clarity.
            import base64, os
            def load_icon(name: str) -> str:
                """Load an icon image from the assets folder or local directory and return a data URI.

                Icons are stored in the `assets` subdirectory relative to this
                script.  If the file is not found there, we fall back to
                loading from the current directory.  The image is encoded
                as a base64 data URI so that it can be embedded directly
                into the HTML/JS without separate file requests.

                Parameters
                ----------
                name : str
                    Filename of the icon (e.g. ``'kiosk.jpg'``).

                Returns
                -------
                str
                    A data URI string representing the loaded image.
                """
                # Determine the directory of this script
                script_dir = os.path.dirname(__file__)
                # Path to the assets directory
                asset_dir = os.path.join(script_dir, 'assets')
                # Try to load from assets directory first
                candidate_paths = [
                    os.path.join(asset_dir, name),
                    os.path.join(script_dir, name),
                ]
                for path in candidate_paths:
                    if os.path.isfile(path):
                        with open(path, 'rb') as f:
                            data = base64.b64encode(f.read()).decode('utf-8')
                        # Determine MIME type based on file extension
                        ext = os.path.splitext(path)[1].lower()
                        if ext == '.png':
                            mime = 'image/png'
                        else:
                            mime = 'image/jpeg'
                        return f"data:{mime};base64,{data}"
                # If the file is not found, return an empty data URI to avoid errors
                return ''

            # Include all icons for each resource type.  In addition to the
            # existing station icons, we add a customer icon (customer.png)
            # which will replace the blue dot in the animation.  The
            # load_icon helper detects file extensions and returns an
            # appropriate MIME type (image/png vs image/jpeg).
            icon_data = {
                # Use only PNG versions of the icons.  JPEG files are located
                # in ``assets/jpegs`` and are deliberately ignored.
                'kiosk': load_icon('kiosk.png'),
                'register': load_icon('register.png'),
                'chef': load_icon('chef.png'),
                'expo': load_icon('expo.png'),
                'drink_station': load_icon('drink_station.png'),
                'condiment_station': load_icon('condiment_station.png'),
                'door': load_icon('door.png'),
                'table': load_icon('table.png'),
                # Use customer2.png for the customer icon.  We ignore customer.png.
                'customer': load_icon('customer2.png'),
            }
            # Serialise the icon data for JavaScript.  We convert the keys to
            # JSON so they can be referenced directly in the JS code.  Note
            # that exit shares the door icon; this is handled in the JS.
            icon_data_str = json.dumps(icon_data)

            # Prepare JSON strings for frames, queues, busies and node info
            frames_json = json.dumps(frames_list)
            queue_json_str = json.dumps(queue_json)
            busy_json_str = json.dumps(busy_json)
            node_positions_str = json.dumps(node_positions)
            node_colors_str = json.dumps(node_colors)
            node_labels_str = json.dumps(node_labels)
            busy_caps_str = json.dumps(busy_caps)
            # Canvas size: height only; width is determined by Streamlit container
            # Increase the canvas height to provide more vertical space for
            # stations and their labels.  A larger height helps ensure
            # stations like the kiosk, register and exit are not cut off when
            # displayed in the browser.
            # Height of the p5.js canvas.  A taller canvas ensures that
            # stations and queue bars fit comfortably in the vertical space.
            # Height of the p5.js canvas.  Increase this to provide more
            # vertical room so stations like kiosks, registers and exit are
            # fully visible without being cut off.  A larger canvas height
            # improves the visual spacing of stations and queues.
            canvas_height = 800
            # Build JS and HTML
            js_template = """
            <script src="https://cdn.jsdelivr.net/npm/p5@1.4.2/lib/p5.min.js"></script>
            <div id="p5-container"></div>
            <script>
            const frames = {frames_json};
            const queueSeries = {queue_json};
            const busySeries = {busy_json};
            const nodePositions = {node_positions};
            const nodeColors = {node_colors};
            const nodeLabels = {node_labels};
            const busyCaps = {busy_caps};
            // Icon data URIs encoded in base64.  Each key corresponds to a
            // resource type.  Exit uses the same image as door and will be
            // mapped accordingly in preload.
            const imgData = {icon_data};
            // Initialise icons dictionary. Use double braces to avoid Python
            // str.format interpreting this as a formatting placeholder. These
            // braces will render as a single pair in the final JavaScript.
            const icons = {{}};
            const fps = {fps_value};
            const canvasH = {canvas_h};
            let frameIndex = 0;
            let isPaused = false;
            const sketch = (p) => {{
              p.preload = () => {{
                // Load icons from the provided data URIs.  We map plural
                // resource prefixes to singular icon keys.  Exit shares the
                // door icon.  If a key is missing, no icon will be drawn.
                icons['kiosks'] = p.loadImage(imgData.kiosk);
                icons['registers'] = p.loadImage(imgData.register);
                icons['cooks'] = p.loadImage(imgData.chef);
                icons['expo'] = p.loadImage(imgData.expo);
                icons['drinks'] = p.loadImage(imgData.drink_station);
                icons['condiments'] = p.loadImage(imgData.condiment_station);
                icons['tables'] = p.loadImage(imgData.table);
                icons['door'] = p.loadImage(imgData.door);
                icons['exit'] = icons['door'];
                // Load customer icon separately.  If the customer icon is
                // missing, no image will be drawn and a fallback will be
                // used instead when rendering customers.
                icons['customer'] = p.loadImage(imgData.customer);
              }};
              p.setup = () => {{
                // Create a canvas that spans the full available width.  We no
                // longer subtract a fixed margin because Streamlit will
                // automatically handle horizontal padding.  Leaving the
                // width unadjusted ensures that stations on the far left and
                // right are visible and not cut off.
                p.createCanvas(p.windowWidth, canvasH);
                // Disable smoothing in p5.js to keep images sharp.  See p5.js
                // documentation for noSmooth(): https://p5js.org/reference/#/p5/noSmooth
                p.noSmooth();
                // Disable image smoothing on the underlying 2D context so that
                // scaled icons remain crisp.  Without this, the icons will
                // appear blurry when drawn at a smaller size.  See MDN
                // documentation for imageSmoothingEnabled: https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/imageSmoothingEnabled
                if (p.drawingContext && p.drawingContext.imageSmoothingEnabled !== undefined) {{
                  p.drawingContext.imageSmoothingEnabled = false;
                }}
                p.frameRate(fps);
              }};
              p.windowResized = () => {{
                p.resizeCanvas(p.windowWidth, canvasH);
              }};
              p.draw = () => {{
                p.background(255);
                const scaleX = p.width / 10.0;
                const scaleY = canvasH / 4.0;
                const rectW = 50;
                const rectH = 32;
                // Draw stations with icons and labels.  Each node key is of
                // the form "resource_index" (e.g. "kiosks_0").  We derive
                // the resource prefix to look up the correct icon.  If an
                // icon is missing for a given prefix, nothing will be drawn.
                // Define station icon height (fixed).  The width will be computed
                // dynamically from the inherent aspect ratio of the image to avoid
                // distortion.  A larger height makes icons clearer.
                const stationIconH = 100;
                // Define customer icon height (fixed).  Customer width will be computed
                // from the inherent aspect ratio of the customer image.
                const customerIconH = 80;
                for (const key in nodePositions) {{
                  const pos = nodePositions[key];
                  const x = pos[0] * scaleX;
                  const y = pos[1] * scaleY;
                  // Determine resource prefix (e.g. "kiosks", "registers")
                  let prefix = key;
                  if (key.includes('_')) {{
                    prefix = key.split('_')[0];
                  }}
                  const iconImg = icons[prefix] || null;
                  if (iconImg) {{
                    // Compute the display width using the original aspect ratio.
                    const ratio = iconImg.width / iconImg.height;
                    const iconW = ratio * stationIconH;
                    const iconH = stationIconH;
                    // Draw the icon centred horizontally and slightly above
                    // the y position so that the label can fit below.
                    p.image(iconImg, x - iconW/2, y - iconH/2 - 5, iconW, iconH);
                  }} else {{
                    // Fallback: draw a pastel rectangle if no icon
                    const fillCol = nodeColors[key] || '#cccccc';
                    p.fill(p.color(fillCol));
                    p.stroke(180);
                    p.rect(x - rectW/2, y - rectH/2, rectW, rectH, 5);
                  }}
                  // Draw the label below the icon/rectangle.  The vertical
                  // offset uses the station icon height so that text always
                  // appears below the image regardless of its aspect ratio.
                  p.noStroke();
                  p.fill(50);
                  p.textSize(10);
                  p.textAlign(p.CENTER, p.TOP);
                  const label = nodeLabels[key] || key;
                  p.text(label, x, y + stationIconH/2 + 2);
                }}
                // Draw queue squares to the left of the first station in each group
                const queueSpacing = 0.15;
                const queueSize = 8;
                for (const resKey in queueSeries) {{
                  const qlen = queueSeries[resKey][frameIndex] || 0;
                  if (qlen <= 0) continue;
                  // find representative station key for this resource
                  let repKey = null;
                  for (const nk in nodePositions) {{
                    if (nk.startsWith(resKey + '_')) {{
                      repKey = nk;
                      break;
                    }}
                  }}
                  if (!repKey) continue;
                  const basePos = nodePositions[repKey];
                  const baseX = basePos[0] * scaleX;
                  const baseY = basePos[1] * scaleY;
                  for (let i = 0; i < qlen; i++) {{
                    const offset = (i + 1) * queueSpacing * scaleX;
                    const x = baseX - rectW/2 - offset - queueSize;
                    // Draw the queue square with a black outline for better visibility.
                    p.fill(p.color(nodeColors[repKey] || '#bbbbbb'));
                    p.stroke(0);
                    p.strokeWeight(1);
                    p.rect(x, baseY - queueSize / 2, queueSize, queueSize);
                    p.noStroke();
                  }}
                }}
                // Draw busy counters (top left)
                const br = busySeries['registers'][frameIndex] || 0;
                const bc = busySeries['cooks'][frameIndex] || 0;
                const be = busySeries['expo'][frameIndex] || 0;
                const busyText = 'Registers: ' + Math.floor(br) + '/' + busyCaps['registers'] + '   Cooks: ' + Math.floor(bc) + '/' + busyCaps['cooks'] + '   Expo: ' + Math.floor(be) + '/' + busyCaps['expo'];
                p.fill(50);
                p.textSize(12);
                p.textAlign(p.LEFT, p.TOP);
                p.text(busyText, 10, 10);
                // Draw customers (moving dots)
                const positions = frames[frameIndex] || [];
                for (const pos of positions) {{
                    const cx = pos[0] * scaleX;
                    const cy = pos[1] * scaleY;
                    const custImg = icons['customer'];
                    if (custImg) {{
                        // Compute customer display width using aspect ratio.
                        const ratio = custImg.width / custImg.height;
                        const customerW = ratio * customerIconH;
                        const customerH = customerIconH;
                        // Draw the customer icon centred at the position
                        p.image(custImg, cx - customerW / 2, cy - customerH / 2, customerW, customerH);
                    }} else {{
                        // Fallback: draw a blue circle if no image
                        p.fill(0, 102, 204);
                        p.noStroke();
                        p.ellipse(cx, cy, 10, 10);
                    }}
                }}
                if (!isPaused) {{
                  frameIndex++;
                  if (frameIndex >= frames.length) {{
                    frameIndex = frames.length - 1;
                    isPaused = true;
                  }}
                }}
              }};
            }};
            new p5(sketch, document.getElementById('p5-container'));
            </script>
            <div style="margin-top:10px;">
              <button id="pauseBtn">Pause</button>
              <button id="skipBtn">Skip</button>
            </div>
            <script>
            const pauseBtn = document.getElementById('pauseBtn');
            const skipBtn = document.getElementById('skipBtn');
            pauseBtn.addEventListener('click', () => {{
              isPaused = !isPaused;
              pauseBtn.innerText = isPaused ? 'Resume' : 'Pause';
            }});
            skipBtn.addEventListener('click', () => {{
              frameIndex = frames.length - 1;
              isPaused = true;
              pauseBtn.innerText = 'Resume';
            }});
            </script>
            """
            js_code = js_template.format(
                frames_json=frames_json,
                queue_json=queue_json_str,
                busy_json=busy_json_str,
                node_positions=node_positions_str,
                node_colors=node_colors_str,
                node_labels=node_labels_str,
                busy_caps=busy_caps_str,
                fps_value=fps,
                canvas_h=canvas_height,
                icon_data=icon_data_str,
            )
        # Store results in session state so they persist on interactions
        st.session_state['results'] = results
        st.session_state['util_data'] = pd.DataFrame({
            'Resource': ['Kiosk', 'Register', 'Cook', 'Expo', 'Drink', 'Condiment', 'Tables'],
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
        st.session_state['qlen_data'] = pd.DataFrame({
            'Resource': ['Kiosk', 'Register', 'Cook', 'Expo', 'Drink', 'Condiment', 'Tables'],
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
        # After simulation and frame generation, render results and animation in tabs
        with tab_animation:
            if skip_animation:
                # Skip entire animation if selected in sidebar
                st.warning(
                    "Animation skipped. Please open the 'Summary Results' tab to view metrics."
                )
            else:
                # Always display the animation when not skipping via sidebar.  Width is omitted
                # so the component expands to the maximum available space.
                st.components.v1.html(js_code, height=canvas_height + 50, scrolling=False)
        # Always display summary results in its tab after a run
        with tab_summary:
            st.subheader("Summary Results")
            # Retrieve stored results
            res = st.session_state.get('results', None)
            if res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Customers served", res["served"])
                c2.metric(
                    "Avg time in system (min)",
                    f"{res['avg_time_min']:.2f}" if res['served'] > 0 else "NA"
                )
                c3.metric(
                    "90th percentile time (min)",
                    f"{res['p90_time_min']:.2f}" if res['served'] >= 10 else "NA"
                )
                st.subheader("Resource Utilisation")
                st.bar_chart(st.session_state['util_data'].set_index('Resource'))
                st.subheader("Average Queue Lengths")
                st.bar_chart(st.session_state['qlen_data'].set_index('Resource'))
            else:
                st.info("Run the simulation to see summary results.")
    # If no run has occurred yet, present placeholder in summary tab
    if not run_button and 'results' not in st.session_state:
        with tab_summary:
            st.info("Run the simulation to see summary results.")


if __name__ == "__main__":
    main()