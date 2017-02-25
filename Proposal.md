Fast placement using convex optimization
===

Team Members: Kevin Kiningham

Project Goal
---
The goal of placement in hardware design is to map a given netlist (represented
as a graph) to a physical location on a die or FPGA (usually represented as a
2d coordinate).
Typically, placement also tries to minimize some notion of power or delay (given
as a function of the distance between adjacent gates in the netlist) as well as
congestion (the density of gates in a given area on chip).
Additionally, the placement may be constrained by requiring that gate placement
does not overlap, or by fixing the position of nodes ahead of time.

This is a complicated optimization problem, and placement in practice is a
difficult and time consuming step of hardware development.

This project proposes to use methods from convext optimization to solve large
placement problems quickly.
I plan to approximate the placement problem as convex problem (including
constraints) over real valued placements and then use
[branch and bound](https://web.stanford.edu/class/ee364b/lectures/bb_slides.pdf)
to find an optimal (or near optimal) solution over the integers.
This works well in practice for many other optimization problems and can be
very fast even with a large number of constraints and variables.

Deliverables
---
I envision two major deliverables
    - A program that can take in some form of netlist and produce a placement.
    - A comparison between my solution and some other solution (i.e. simulated annealing)

Note that the final solution is not intended to be compeative with cutting
edge research or commertial products.
My goal is to (hopefully) show that branch and bound + convex optimization is a
promising direction for future research on placement problems.

Challenges
---
Optimal placement is not convex, even over the reals.
However, most placement tools today already optimize using approximations, some
of which are convex.
For example, approximating the total wire length as the sum of the bounding box
around cliques of gates in the netlist is convex and fairly close to the
[actual wire length in practice](http://dl.acm.org/citation.cfm?id=1112348).
For this project I plan on optimizing using the convex approximations, and thus
the quality of the placement depends on how good thse approximations are.

Additionally, it's not clear how fast branch and bound will be for this particular
problem, even if it may be fast for other problems.

Previous Work
---
The ideas here draw heavily from work on using quadratic programming for
pacement, which uses a quadratic approximation for wire lengths.
See [slides 29-49](http://vlsicad.eecs.umich.edu/KLMH/downloads/book/chapter4/chap4-111206.pdf)
for examples.

Resources
---
I plan to use [CVXPY](http://www.cvxpy.org/en/latest/index.html)
to do the actual optimization.

Major results
---
	- Solve a small hard coded example without branch and bound
	- Solve a non-hard coded example without branch and bound
	- Solve using branch and bound
    - Compare timing/results with other methods (SMT or simulated annealing)

Timeline
---
    - (2/27) Install CVXPy and familierize myself with it (I've only used CVX before)
    - (3/1) Get small hard coded non branch-and-bound example working.
    - (3/7) Be able to read and solve non branch-and-bound examples from a file.
    - (3/14) Finish branch and bound
    - (3/20) Finish comparison with simulated annealing, other methods
    - (3/21) Finish paper/presentation
