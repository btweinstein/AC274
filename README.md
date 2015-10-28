# AC274
Exercises for Sauro Succi's computational fluid dynamics course at harvard.

## 1d Advection-Diffusion-Reaction (ADR) Solver

![Solution](https://github.com/btweinstein/AC274/blob/master/examples/1d_adr_example.png)
![Solution history](https://github.com/btweinstein/AC274/blob/master/examples/1d_adr_solution_history.png)

## 2d Advection-Diffusion Reaction (ADR) Solver

Note that although the code seems to work for short times, it appears to somewhat unstable. If you have a better
way of doing this with finite difference, let me konw!

![2d adr flow field](https://github.com/btweinstein/AC274/blob/master/examples/2d_example_flow.png)
We create a flow field. How does an advection-reaction-diffusion system propagate through this? We solve

$$\frac{df}{dt}=-(\vec{v}\cdot \nabla)f + D \nabla^2 f + sf(1-f)$$

in it for a given choice of parameters and and initial conditions and find something like the below.

![Advection Diffusion in the flow](https://github.com/btweinstein/AC274/blob/master/examples/2d_advection_diffusion_in_flow.png)

The solution looks like what we would expect intuitively.

See the below movie for the evolution of this plot. I played with a number of parameters but rather like the below movie.

[Movie](https://github.com/btweinstein/AC274/blob/master/examples/smaller_s_and_D%20kept%20stack.avi "Movie")
