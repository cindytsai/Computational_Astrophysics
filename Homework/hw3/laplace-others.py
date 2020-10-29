import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d


# Settings
L = 1.0
N = 16  # TODO
dx = L / (N + 1)


def Error(phi_result, lo, dx):

    error = 0.0

    for i in range(1, phi_result.shape[0] - 1):
        for j in range(1, phi_result.shape[1] - 1):
            residual = dx**-2 * (phi_result[i + 1, j] + phi_result[i - 1, j] + phi_result[i, j + 1] + phi_result[i, j - 1] - 4.0 * phi_result[i, j]) - lo[i, j]
            error = error + abs(residual)

    error = error / ((phi_result.shape[0] - 2.0) * (phi_result.shape[1] - 2.0))

    return error


def Jacobi(phi, lo, dx, target_error, iter_max):

    err = target_error + 1.0
    phi_result = np.copy(phi)
    iter = 0

    while (err > target_error and iter < iter_max):

        iter = iter + 1

        for i in range(1, phi.shape[0] - 1):
            for j in range(1, phi.shape[1] - 1):
                phi_result[i, j] = 0.25 * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1] - dx**2 * lo[i, j])

        err = Error(phi_result, lo, dx)
        phi = np.copy(phi_result)
        print(iter, ":", err)

    return iter, err, phi


def GaussSeidel(phi, lo, dx, target_error, iter_max):

    err = target_error + 1.0
    phi_result = np.copy(phi)
    iter = 0

    while (err > target_error and iter < iter_max):

        iter = iter + 1

        for i in range(1, phi.shape[0] - 1):
            for j in range(1, phi.shape[1] - 1):
                phi_result[i, j] = 0.25 * (phi_result[i - 1, j] + phi[i + 1, j] + phi_result[i, j - 1] + phi[i, j + 1] - dx**2 * lo[i, j])

        err = Error(phi_result, lo, dx)
        phi = np.copy(phi_result)
        print(iter, ":", err)

    return iter, err, phi


def SOR(phi, lo, omega, dx, target_error, iter_max):

    err = target_error + 1.0
    phi_result = np.copy(phi)
    iter = 0

    while(err > target_error and iter < iter_max):

        iter = iter + 1

        for i in range(1, phi.shape[0] - 1):
            for j in range(1, phi.shape[1] - 1):
                phi_result[i, j] = phi[i, j] + 0.25 * omega * (phi_result[i - 1, j] + phi[i + 1, j] + phi_result[i, j - 1] + phi[i, j + 1] - 4.0 * phi[i, j] - dx**2 * lo[i, j])

        err = Error(phi_result, lo, dx)
        phi = np.copy(phi_result)
        print(iter, ":", err)

    return iter, err, phi



# Create (x, y) coordinate, (0,0) starts at the bottom left
x = np.linspace(0, 1, N + 2, endpoint=True, dtype=np.float64)
y = np.linspace(0, 1, N + 2, endpoint=True, dtype=np.float64)
y = np.flip(y, axis=0)
x, y = np.meshgrid(x, y)

# Initialize the boundary condition
phi = np.zeros((N + 2, N + 2), dtype=np.float64)
# lo = 2.0 * x * (y - 1) * (y - 2.0 * x + x * y + 2) * np.exp(x - y)
lo = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Run the algorithm
print("Running...")
target_error = 1.0e-10
iter_max = 1000000
start_time = time.time()
result = GaussSeidel(phi, lo, dx, target_error, iter_max)  # TODO
# omega = 1.5
# result = SOR(phi, lo, omega, dx, target_error, iter_max)
end_time = time.time()

# Result
print("iter = %d" % result[0])
print("err  = %.23f" % result[1])
print("time used = %.5f sec" % (end_time - start_time))
print("phi = ")
print(result[2])

# Calculate the analytic solution
ans = - 0.5 / 2.0 / 2.0 / np.pi / np.pi * np.copy(lo)

# Compare with the result
error = np.sum(np.absolute(result[2] - ans)) / np.power(N, 2, dtype=np.float64)
print("Error : %.5e" % error)

# Plot the final result
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title("Final Result", fontsize=14)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
surf = ax.plot_surface(x, y, ans, cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
