import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

# I] Setup the system of equations symbolically using sympy
# Define the symbols
t, g = smp.symbols('t g')  # t for time and g for gravity
m1, m2 = smp.symbols('m1 m2')  # Mass of the 2 balls of the pendulum
L1, L2 = smp.symbols('L1, L2')  # Length of the 2 arms of the pendulum

# Defining the two functions of time theta 1 and theta 2
the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
# Explicitly making them functions of time
the1 = the1(t)
the2 = the2(t)

# Defining the first derivatives (fd) and second derivatives (sd) of the newly made functions of time
the1_fd = smp.diff(the1, t)
the2_fd = smp.diff(the2, t)
the1_sd = smp.diff(the1_fd, t)
the2_sd = smp.diff(the2_fd, t)

# Defining the variables of pendulum's balls position
x1 = L1*smp.sin(the1)
y1 = -L1*smp.cos(the1)
x2 = L1*smp.sin(the1)+L2*smp.sin(the2)
y2 = -L1*smp.cos(the1)-L2*smp.cos(the2)
# Thanks to the cartesian coordinates we can get the Kinetic energy
T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T = T1 + T2
# Thanks to the cartesian coordinates we can also get the Potential energy
V1 = m1*g*y1
V2 = m1*g*y2
V = V1 + V2
# Which mean we can now have the Lagrangian
L = T-V
# So we can then obtain the Lagrange's equations of motion (which we simplify to be sure the equations remains clean)
LEM1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_fd), t).simplify()
LEM2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_fd), t).simplify()

# II] Once we have those equations we can solve them numerically
# Getting the solutions
sols = smp.solve([LEM1, LEM2], (the1_sd, the2_sd), simplify=False, rational=False)

# Thanks to those solutions we can get our numerical functions
dz1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_fd,the2_fd), sols[the1_sd])  # First argument is the variables on which the function's depends on, and the second is the symbolic expression of the function.
dz2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_fd,the2_fd), sols[the1_sd])
dthe1dt_f = smp.lambdify(the1_fd, the1_fd)
dthe2dt_f = smp.lambdify(the2_fd, the2_fd)


# Making a function to calculate the time derivatives of the pendulum's state variables (angles and angular velocities)
def dSdt(S, t, g, m1, m2, L1, L2):
    the1, z1, the2, z2 = S
    return [
        dthe1dt_f(z1),
        dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
        dthe2dt_f(z2),
        dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
    ]


# Now we can solve the system of ODE using a solver
t = np.linspace(0, 40, 1001)  # Solving it between 0 and 40 second for 1001 different time points (25 frames per seconds)
g = 9.81
m1 = 2
m2 = 1
L1 = 2
L2 = 1
ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t, args=(g,m1,m2,L1,L2))  # The 1001 answers to the system of equation


# Declare a function which calculates the x and y coordinates of the two pendulum masses based on their angles and lengths
def get_x1y1x2y2(t, the1, the2, L1, L2):
    return (L1*np.sin(the1),
            -L1*np.cos(the1),
            L1*np.sin(the1) + L2*np.sin(the2),
            -L1*np.cos(the1) - L2*np.cos(the2))


x1, y1, x2, y2 = get_x1y1x2y2(t, ans.T[0], ans.T[2], L1, L2)  # All the positions of the pendulum's masses for 1001 frames in 40s of time


# III] Then we can animate the solution
def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])  # 3 points here, 0=origin, 1=first ball, 2=second ball


# Setup matplotlib axis
fig, ax = plt.subplots(1,1, figsize=(8, 8))
ax.set_facecolor('white')
ax.get_xaxis().set_ticks([])  # Hide x axis ticks
ax.get_yaxis().set_ticks([])  # Hide y axis ticks
# Drawing the pendulum
ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
ax.set_ylim(-4, 4)
ax.set_xlim(-4, 4)
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('pendulum_animation.gif', writer='pillow', fps=25)
