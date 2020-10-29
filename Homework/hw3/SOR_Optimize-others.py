import numpy as np
import scipy.optimize as opt
import time

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


def SOR(phi, lo, omega, dx, target_error, iter_max):

    print("omega:", omega)

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
L = 1.0
N = 64
k = 2.0 * np.pi / N
x = np.arange(0, N)
y = np.arange(0, N)
xx, yy = np.meshgrid(x, y)

# Initialize the boundary condition
phi = np.zeros(N * N)
rho = np.zeros((N + 2) * (N + 2))
boundary = ((yy % N == 0) | (yy % N == N - 1) | (xx % N == 0) | (xx % N == N - 1)).ravel()
phi_analytic = np.exp(-k * xx) * np.sin(k * yy)
phi[boundary] = phi_analytic.ravel()[boundary]

phi_i = np.zeros(phi.shape[0] + 2, phi.shape[1] + 2)
phi_i[1:phi.shape[0] - 1, 1:phi.shape[1] - 1] = phi

# Optimize SOR
print("Optimize omega...")
target_error = 1.0e-10
iter_max = 10000


def fcn(omega):
    phi = np.zeros((N + 2, N + 2), dtype=np.float64)
    lo = 2.0 * x * (y - 1) * (y - 2.0 * x + x * y + 2) * np.exp(x - y)
    result = SOR(phi, lo, omega, dx, target_error, iter_max)
    return result[0]


omega = 1.5
optimize = opt.minimize(fcn, omega, method='SLSQP', bounds=[(1.01, 1.99)], options={'eps': 0.01})
omega = optimize.x

# Run the algorithm
print("Running...")
start_time = time.time()
result = SOR(phi, lo, omega, dx, target_error, iter_max)
end_time = time.time()


# Result
print("Optimal omega : ", omega)
print("iter = %d" % result[0])
print("err  = %.23f" % result[1])
print("time used = %.5f sec" % (end_time - start_time))
print("phi = ")
print(result[2])

# Calculate the analytic solution
ans = np.exp(-k * xx) * np.sin(k * yy)

# Compare with the result
error = np.sum(np.absolute(result[2] - ans)) / np.power(N, 2, dtype=np.float64)
print("Error : %.5e" % error)
