# AC274
Exercises for Sauro Succi's computational fluid dynamics course at harvard.

## 1d Advection-Diffusion-Reaction (ADR) Solver

![Solution](https://github.com/btweinstein/AC274/blob/master/examples/1d_adr_example.png)
![Solution history](https://github.com/btweinstein/AC274/blob/master/examples/1d_adr_solution_history.png)

## 2d Advection-Diffusion Reaction (ADR) Solver

Note that although the code seems to work for short times, it appears to become unstable at long times. If anyone is able to diagnose the source of the problem, let me know!

![2d adr flow field][https://github.com/btweinstein/AC274/blob/master/examples/2d_example_flow.png]
We create a flow field. How does an advection-reaction-diffusion system propagate through this? We solve

$$\frac{df}{dt}=-(\vec{v}\cdot \nabla)f + D \nabla^2 f + sf(1-f)$$

in it for a given choice of parameters and and initial conditions and find something like the below.

![Advection Diffusion in the flow][https://github.com/btweinstein/AC274/blob/master/examples/2d_advection_diffusion_in_flow.png]

This is a nice looking solution, but don't trust it too much right now as my code appears to be unstable for the 2d case.
