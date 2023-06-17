import datetime
import random
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import names
import numpy as np
import pandas as pd
from matplotlib import colors
from mip import BINARY, Model, minimize, xsum

random.seed(999)

# ---------- #
# -- Sets -- #
# ---------- #

# Set of CSRs
n_csr_names: int = 10
csr_names: List[str] = [names.get_full_name() for _ in range(n_csr_names)]

for i in csr_names:
    print(i)


# Set of days
n_days: int = 7
days: List[datetime.date] = [
    datetime.date.today() + datetime.timedelta(days=v) for v in range(1, n_days + 1)
]

for d in days:
    print(d)

# ---------------- #
# -- Parameters -- #
# ---------------- #

# CSR availability
csr_availability: np.array = np.random.choice(
    [0, 1], size=(n_csr_names, n_days), p=[0.2, 0.8]
)

print(pd.DataFrame(csr_availability, index=csr_names, columns=days))

# Daily estimates of customer volumes; normal distribution with mean 120 and std 10
daily_customer_volumes: np.array = np.array(
    [round(v) for v in np.random.normal(loc=120, scale=10, size=7)]
)
# Handle potential negative values
daily_customer_volumes[daily_customer_volumes < 0] = 0

print(pd.DataFrame(daily_customer_volumes, index=days, columns=["Customer volume"]))

# Average number of customers a CSR can handle in a day
csr_productivity: int = 16

# ------------------ #
# -- Define Model -- #
# ------------------ #

# -- Model --
model = Model("staff-scheduling-part-1")

# -- Sets --

# Create sets of integers, since Python-MIP cannot deal with other data types, i.e. `str` or `datetime.date`
D: Set[int] = set(range(n_days))
I: Set[int] = set(range(n_csr_names))

# -- Decision variables --

# Create decision variables; specify type to BINARY
x = [[model.add_var(var_type=BINARY) for d in D] for i in I]

# Create derived variables for daily shortages; use default type CONTINUOUS
y = [model.add_var() for d in D]

# -- Objective function --

# Minimize the sum of the daily shortages
model.objective = minimize(xsum(y[d] for d in D))

# -- Constraints --

# Derived variables constraints
for d in D:
    model += y[d] >= 0
    model += y[d] >= daily_customer_volumes[d] - csr_productivity * xsum(
        x[i][d] for i in I
    )

# CSR availability constraints
for i, d in zip(I, D):
    model += x[i][d] <= csr_availability[i][d]

# 5-day workweek constraints
for i in I:
    model += xsum(x[i][d] for d in D) <= 5

# Decision variables constraints are omitted since we specify the type to BINARY

# ----------------- #
# -- Solve Model -- #
# ----------------- #

# Run the model
status = model.optimize(max_seconds=5)

print(f"The model solution is {status.name}.")

# ------------- #
# -- Results -- #
# ------------- #

# Get the objective value, i.e. the total customer shortage
total_customer_shortage: int = int(model.objective_value)

# Calculate the total customer volume
total_customer_volume: int = daily_customer_volumes.sum()

print(f"Total customer volume: {total_customer_volume}")
print(
    "Total customer shortage: "
    f"{total_customer_shortage} ({round(100 * total_customer_shortage / total_customer_volume, 2)}% of total volume)"
)

# Calculate the daily customer shortages
daily_customer_shortages: List[int] = [int(y[d].x) for d in D]

# Plot customer volume and shortage
fig, ax = plt.subplots(figsize=(18, 4), dpi=100)
ax.plot_date(days, daily_customer_volumes, "o--", color="#5fbb7d", label="Daily volume")
ax.plot_date(
    days, daily_customer_shortages, "o--", color="#666666", label="Daily shortage"
)

# Add label, grid and legend
plt.ylabel("Customers")
plt.grid(alpha=0.25)
plt.legend()

fig.tight_layout()
plt.savefig(fname="results/part_1_customer_shortage.png")

# Get decision variables, i.e. the CSR schedule
csr_schedule = np.array([[x[i][d].x for d in D] for i in I])

_SHIFT_LABELS: Dict[int, str] = {
    1: "09:00-18:00" + "\n" + "(break 13:00-14:00)",
}

# Define color coding for the 2D raster
color_map: colors.ListedColormap = colors.ListedColormap(["#eeeeee", "#5fbb7d"])
bounds: List[float] = [0, 1]
norm: colors.BoundaryNorm = colors.BoundaryNorm(bounds, color_map.N)

# Plot raster
fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
ax.imshow(csr_schedule, cmap=color_map, norm=norm, aspect="auto")

# Add text in the tiles
for i in I:
    for d in D:
        # If CSR i is selected to work during day d, print the schedule
        if csr_schedule[i][d] == 1:
            text = ax.text(
                d,
                i,
                _SHIFT_LABELS[csr_schedule[i][d]],
                size=8,
                ha="center",
                va="center",
                color="white",
            )
        # If CSR i is not selected to work during day d because we was not available, print "Unavailable"
        elif csr_schedule[i][d] == 0 and csr_availability[i][d] == 0:
            text = ax.text(
                d, i, "Unavailable", size=8, ha="center", va="center", color="black"
            )

# Remove major ticks
plt.tick_params(axis="x", which="both", bottom=False)
plt.tick_params(axis="y", which="both", left=False)

# Add labels in the major tick
plt.xticks(np.arange(len(D)), days, size=9)
plt.yticks(np.arange(len(I)), csr_names, size=9)

# Plot grid on minor axes
ax.set_xticks([x - 0.5 for x in range(1, len(D))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(I))], minor=True)
plt.grid(which="minor", ls="-", lw=2, color="white")

# Remove frame
ax.set(frame_on=False)

fig.tight_layout()
plt.savefig(fname="results/part_1_csr_schedule.png")
