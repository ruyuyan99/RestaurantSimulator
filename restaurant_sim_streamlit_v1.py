"""
Restaurant Simulation Streamlit App
=================================

This Streamlit app provides an interactive interface to run and visualize a
discrete event simulation of a fast casual restaurant.  It is based on the
simulation logic defined in the accompanying animation script but omits
graphical animation in favour of numerical analysis and charts.  The user
can adjust parameters such as the arrival rate and resource capacities,
execute the simulation for a specified duration, and review summary metrics
and resource utilization in a clear, modern web interface.

Usage
-----
Install the required packages (``salabim``, ``streamlit``, and
``pandas``) and run this script with Streamlit::

    streamlit run restaurant_sim_streamlit.py

You can customise the parameters in the sidebar and press the **Run
Simulation** button to generate new results.
"""

import random
import statistics
from typing import List, Tuple

import pandas as pd
import streamlit as st
import salabim as sim


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
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
D = [
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

    A small random congestion factor (1.0–1.25) is applied to the base time,
    which is the distance divided by walking speed.  This mirrors the behaviour
    used in the original animation script.
    """
    base = D[a][b] / WALK_SPEED_MPS
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


# -----------------------------------------------------------------------------
# Simulation components
# -----------------------------------------------------------------------------
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
) -> dict:
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
    results = {}
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
        # Mean number of claimers over time
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


# -----------------------------------------------------------------------------
# Streamlit interface
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Restaurant DES", layout="wide")
    st.title("🍽️ Restaurant Discrete Event Simulation")
    st.write(
        "This interactive app simulates a fast casual restaurant. Adjust the\n"
        "parameters in the sidebar and click **Run Simulation** to see how\n"
        "changes affect throughput, waiting times, and resource utilization."
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

    run_button = st.sidebar.button("Run Simulation")

    if run_button:
        with st.spinner("Running simulation, please wait..."):
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

        st.write(
            "The simulation model is stochastic; running it multiple times may\n"
            "yield slightly different results. Adjust the parameters and click\n"
            "**Run Simulation** again to explore different scenarios."
        )


if __name__ == "__main__":
    main()