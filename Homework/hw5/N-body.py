import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# constants
G = 1.0       # gravitational constant
dt = 1.0e-2   # time interval for data update

# Initial condition and Settings
#################################
ALGORITHM = {"KDK": (1, "KDK"), "DKD": (2, "DKD"), "RK4": (3, "RK4")}
use_algorithm = ALGORITHM["RK4"]  # TODO
t = 0.0

# Set up each particles initial conditions, (Mass, (x,y), (vx, vy))
# particles = {"0": (2.0, (0, 0), (0, 0)),
#              "1": (0.000000000001, (1.0, 0.0), (0.0, (G * 2.0 / ((1.0**2 + 0.0**2)**0.5))**0.5))}
particles = {"0": (1.0, (-0.97000436, 0.24308753), (0.4662036850, 0.4323657300)),
             "1": (1.0, (0.0, 0.0), (-0.93240737, -0.86473146)),
             "2": (1.0, (0.97000436, -0.24308753), (0.4662036850, 0.4323657300))}
particles = {"0": (1.0, (-0.97000436, 0.24308753), (0.4662036850, 0.4323657300)),
             "1": (1.0, (0.0, 0.0), (-0.93240737, -0.86473146)),
             "2": (1.0, (0.97000436, -0.24308753), (0.4662036850, 0.4323657300)),
             "3": (0.5, (1.0, 1.0), (0.0, 0.0))}
# particles = {"0": (1.0, (0.0, 0.0), (0.0, 0.0)),
#              "1": (3.00e-6, (1.0, 0.0), (0.0, 1.0)),
#              "2": (3.7e-8, (0.0, 2.56e-3), (0.01, 0.0))}
# particles = {"0": (1.0, (1.0, 1.0), (0.0, 0.0)),
#              "1": (1.0, (-1.0, 1.0), (0.0, 0.0)),
#              "2": (1.0, (-1.0, -1.0), (0.0, 0.0)),
#              "3": (1.0, (1.0, -1.0), (0.0, 0.0))}
N = len(particles)

# Assign them to array
mass_arr = np.array([], dtype=np.float64)
x_arr = np.array([], dtype=np.float64)
y_arr = np.array([], dtype=np.float64)
vx_arr = np.array([], dtype=np.float64)
vy_arr = np.array([], dtype=np.float64)

for i in range(N):
    mass_arr = np.append(mass_arr, particles[str(i)][0])
    x_arr = np.append(x_arr, particles[str(i)][1][0])
    y_arr = np.append(y_arr, particles[str(i)][1][1])
    vx_arr = np.append(vx_arr, particles[str(i)][2][0])
    vy_arr = np.append(vy_arr, particles[str(i)][2][1])

# Calculate the Total Energy E0
E0 = (0.5 * mass_arr * (vx_arr**2 + vy_arr**2)).sum()
for i in range(N):
    for j in range(i + 1, N):
        distance = np.sqrt((x_arr[i] - x_arr[j])**2 + (y_arr[i] - y_arr[j])**2)
        potential = - G * mass_arr[i] * mass_arr[j] / distance
        E0 = E0 + potential

# Algorithms
###############################


def DKD(t_start, t_end):
    global t, x_arr, y_arr, vx_arr, vy_arr

    while t < t_end and t >= t_start:

        # Drift
        x_arr = x_arr + vx_arr * 0.5 * dt
        y_arr = y_arr + vy_arr * 0.5 * dt

        # Kicks
        ax_arr = np.zeros(N, dtype=np.float64)
        ay_arr = np.zeros(N, dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # j: source / i: self
                distance = np.sqrt((x_arr[i] - x_arr[j])**2 + (y_arr[i] - y_arr[j])**2)
                a_abs = G * mass_arr[j] / distance**2
                ax_arr[i] = ax_arr[i] + a_abs * (x_arr[j] - x_arr[i]) / distance
                ay_arr[i] = ay_arr[i] + a_abs * (y_arr[j] - y_arr[i]) / distance
        vx_arr = vx_arr + ax_arr * dt
        vy_arr = vy_arr + ay_arr * dt

        # Drift
        x_arr = x_arr + vx_arr * 0.5 * dt
        y_arr = y_arr + vy_arr * 0.5 * dt

        t = t + dt


def KDK(t_start, t_end):
    global t, x_arr, y_arr, vx_arr, vy_arr

    while t < t_end and t >= t_start:

        # Kicks
        ax_arr = np.zeros(N, dtype=np.float64)
        ay_arr = np.zeros(N, dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # j: source / i: self
                distance = np.sqrt((x_arr[i] - x_arr[j])**2 + (y_arr[i] - y_arr[j])**2)
                a_abs = G * mass_arr[j] / distance**2
                ax_arr[i] = ax_arr[i] + a_abs * (x_arr[j] - x_arr[i]) / distance
                ay_arr[i] = ay_arr[i] + a_abs * (y_arr[j] - y_arr[i]) / distance
        vx_arr = vx_arr + ax_arr * 0.5 * dt
        vy_arr = vy_arr + ay_arr * 0.5 * dt

        # Drift
        x_arr = x_arr + vx_arr * dt
        y_arr = y_arr + vy_arr * dt

        # Kicks
        ax_arr = np.zeros(N, dtype=np.float64)
        ay_arr = np.zeros(N, dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # j: source / i: self
                distance = np.sqrt((x_arr[i] - x_arr[j])**2 + (y_arr[i] - y_arr[j])**2)
                a_abs = G * mass_arr[j] / distance**2
                ax_arr[i] = ax_arr[i] + a_abs * (x_arr[j] - x_arr[i]) / distance
                ay_arr[i] = ay_arr[i] + a_abs * (y_arr[j] - y_arr[i]) / distance
        vx_arr = vx_arr + ax_arr * 0.5 * dt
        vy_arr = vy_arr + ay_arr * 0.5 * dt

        t = t + dt


def RK4(t_start, t_end):
    global t, x_arr, y_arr, vx_arr, vy_arr

    t = t_start

    while t < t_end and t >= t_start:

        vx_arr_0 = np.copy(vx_arr)
        vy_arr_0 = np.copy(vy_arr)
        x_arr_0 = np.copy(x_arr)
        y_arr_0 = np.copy(y_arr)

        k1 = np.zeros((4, N, 2), dtype=np.float64)
        k2 = np.zeros((4, N, 2), dtype=np.float64)

        # Run through k[0], k[1], k[2]
        for k in range(3):

            # Update k for velocity
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # j: source / i: self
                    distance = np.sqrt((x_arr[i] - x_arr[j])**2 + (y_arr[i] - y_arr[j])**2)
                    a_abs = G * mass_arr[j] / distance**2
                    k1[k, i, 0] = k1[k, i, 0] + a_abs * (x_arr[j] - x_arr[i]) / distance
                    k1[k, i, 1] = k1[k, i, 1] + a_abs * (y_arr[j] - y_arr[i]) / distance

            # Update k for coordinates
            k2[k, :, 0] = vx_arr
            k2[k, :, 1] = vy_arr

            vx_arr = vx_arr_0 + 0.5 * dt * k1[k, :, 0]
            vy_arr = vy_arr_0 + 0.5 * dt * k1[k, :, 1]
            x_arr = x_arr_0 + 0.5 * dt * k2[k, :, 0]
            y_arr = y_arr_0 + 0.5 * dt * k2[k, :, 1]

        # Run k[3]
        vx_arr = vx_arr_0 + dt * k1[2, :, 0]
        vy_arr = vy_arr_0 + dt * k1[2, :, 1]
        x_arr = x_arr_0 + dt * k2[2, :, 0]
        y_arr = y_arr_0 + dt * k2[2, :, 1]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # j: source / i: self
                distance = np.sqrt((x_arr[i] - x_arr[j])**2 + (y_arr[i] - y_arr[j])**2)
                a_abs = G * mass_arr[j] / distance**2
                k1[3, i, 0] = k1[3, i, 0] + a_abs * (x_arr[j] - x_arr[i]) / distance
                k1[3, i, 1] = k1[3, i, 1] + a_abs * (y_arr[j] - y_arr[i]) / distance

        k2[3, :, 0] = vx_arr
        k2[3, :, 1] = vy_arr

        # Calculate velocity and coordinate through k
        vx_arr = vx_arr_0 + (1.0 / 6.0) * dt * (k1[0, :, 0] + 2.0 * k1[1, :, 0] + 2.0 * k1[2, :, 0] + k1[3, :, 0])
        vy_arr = vy_arr_0 + (1.0 / 6.0) * dt * (k1[0, :, 1] + 2.0 * k1[1, :, 1] + 2.0 * k1[2, :, 1] + k1[3, :, 1])
        x_arr = x_arr_0 + (1.0 / 6.0) * dt * (k2[0, :, 0] + 2.0 * k2[1, :, 0] + 2.0 * k2[2, :, 0] + k2[3, :, 0])
        y_arr = y_arr_0 + (1.0 / 6.0) * dt * (k2[0, :, 1] + 2.0 * k2[1, :, 1] + 2.0 * k2[2, :, 1] + k2[3, :, 1])

        t = t + dt


# Plottings
# ##############################
# plotting parameters
period = 10.0
end_time = 3.0 * period
nstep_per_image = 10
plot_points_num = 10
padded_percent = 0.05

# create figure
fig, ax = plt.subplots(1, 2)
fig.suptitle("%d-body Simulation with " % (N) + use_algorithm[1], fontsize=16)

ax[0].set_xlim(-1.5, +1.5)
ax[0].set_ylim(-1.5, +1.5)
ball, = ax[0].plot([], [], 'ro', ms=10)
text = ax[0].text(0.0, 1.3, '', fontsize=12, color='black', ha='center', va='center')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_aspect('equal')
ax[0].tick_params(top=True, right=True, labeltop=True, labelright=True)
# ax[0].add_artist(plt.Circle((0.0, 0.0), r, color='b', fill=False))

error_plot, = ax[1].plot([], [], 'b.-')
ax[1].set_title("Error=" + r'$\frac{E-E0}{E0}$', fontsize=14)
ax[1].set_ylim(-0.01, 0.01)
ax[1].set_xlim(0.0, dt * plot_points_num)
ax[1].set_xlabel('time')

error_array = np.array([])
time_array = np.array([])

total_time = 0.0


def init():
    ball.set_data([], [])
    text.set(text='')
    error_plot.set_data([], [])

    return ball, text


def update_orbit(ii):
    global t, x_arr, y_arr, vx_arr, vy_arr, error_array, time_array, total_time

    for step in range(nstep_per_image):
        if use_algorithm[0] == 1:
            start_time = time.time()
            KDK(t, t + dt)
            end_time = time.time()
        elif use_algorithm[0] == 2:
            start_time = time.time()
            DKD(t, t + dt)
            end_time = time.time()
        else:
            start_time = time.time()
            RK4(t, t + dt)
            end_time = time.time()

        total_time = total_time + (end_time - start_time)

        if (t >= end_time):
            break

#   calculate energy error
    E = (0.5 * mass_arr * (vx_arr**2 + vy_arr**2)).sum()
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.sqrt((x_arr[i] - x_arr[j])**2 + (y_arr[i] - y_arr[j])**2)
            potential = - G * mass_arr[i] * mass_arr[j] / distance
            E = E + potential

    err = (E - E0) / E0

    error_array = np.append(error_array, err)
    print(err)

    time_array = np.append(time_array, t)

#   update plot
    ball.set_data(x_arr, y_arr)
    text.set(text='t/T = %6.3f, error = %10.3e' % (t / period, err))
    error_plot.set_data(time_array, error_array)

#   plot error settings
    padded = (np.max(error_array) - np.min(error_array)) * padded_percent
    error_plot.axes.set_xlim(np.min(time_array), np.max(time_array))
    error_plot.axes.set_ylim(np.min(error_array) - padded, np.max(error_array) + padded)

#   plot simulation settings
    # if (np.max(x_arr) - np.min(x_arr)) > (np.max(y_arr) - np.min(y_arr)):
    #     padded = (np.max(x_arr) - np.min(x_arr)) * padded_percent
    #     ball.axes.set_xlim(np.min(x_arr) - padded, np.max(x_arr) + padded)
    #     ball.axes.set_ylim(y_arr.sum() - (0.5 * (np.max(x_arr) - np.min(x_arr)) + padded), y_arr.sum() + (0.5 * (np.max(x_arr) - np.min(x_arr)) + padded))
    # else:
    #     padded = (np.max(y_arr) - np.min(y_arr)) * padded_percent
    #     ball.axes.set_ylim(np.min(y_arr) - padded, np.max(y_arr) + padded)
    #     ball.axes.set_xlim(x_arr.sum() - (0.5 * (np.max(x_arr) - np.min(x_arr)) + padded), x_arr.sum() + (0.5 * (np.max(x_arr) - np.min(x_arr)) + padded))

    return ball, text, error_plot


# create movie
nframe = int(np.ceil(end_time / (nstep_per_image * dt)))
anim = animation.FuncAnimation(fig, func=update_orbit, init_func=init,
                               frames=nframe, interval=10, repeat=False)
plt.show()

# Save the error as txt file
np.savetxt("./result/error_" + use_algorithm[1] + ".txt", error_array)
np.savetxt("./result/time.txt", time_array)

print("time used in %s = %f" % (use_algorithm[1], total_time))
