# Bet optimization for Euro2020 Kicktipp

## Assumed scoring rules
The optimization assumes the following rules on kicktipp.de
* Correct match result: 4 poins
* Correct goal difference: 3 points
* Correct tendency/winning team: 2 points
* Results include potential penalty shootouts

## How to run

* cd into repository root
* start `julia --project=.`
* `julia> ] instantiate` # installs required packages
* `julia> using Pluto`
* `julia> Pluto.run()`
* In the new browser window, open the `Simulation.jl` notebook
