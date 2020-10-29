import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

# Settings
sigma = 100.0  # TODO
n_cell = 100  # TODO

# Read data
filename = 'density.dat'
N = 1024
density = np.fromfile(filename, 'float32').reshape(N, N)

# Create Gaussian Filter
x = np.arange(0, n_cell + 1, 1) - (n_cell - 1) / 2.0
x, y = np.meshgrid(x, x)
Gaussian = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
Gaussian = Gaussian / Gaussian.sum()

# Do the convolution with Gaussian Filter
Gaussian_pad0 = np.zeros(density.shape)
Gaussian_pad0[0:Gaussian.shape[0], 0:Gaussian.shape[1]] = Gaussian
Gaussian_pad0 = np.roll(Gaussian_pad0, -(Gaussian.shape[0] // 2), axis=0)
Gaussian_pad0 = np.roll(Gaussian_pad0, -(Gaussian.shape[1] // 2), axis=1)

Gaussian_pad0_k = np.fft.rfft2(Gaussian_pad0)
density_k = np.fft.rfft2(density)
convolution_k = Gaussian_pad0_k * density_k
convolution = np.fft.irfft2(convolution_k)

# Plot the result before and after gaussian filter
plt.imshow(density)
plt.title("Original", fontsize=16)
plt.colorbar()
plt.show()

plt.imshow(convolution)
plt.title(r'$\sigma$' + " = %.1f" % (sigma) + ", cell = %d x %d" % (n_cell, n_cell), fontsize=16)
plt.colorbar()
plt.show()

#####################################
# Find the 2D power spectrum with fft2
# Gaussian_pad0_k_fft2 = np.fft.fft2(Gaussian_pad0)
# density_k_fft2 = np.fft.fft2(density)
# convolution_k_fft2 = Gaussian_pad0_k_fft2 * density_k_fft2
# convolution_k_fft2_PS = np.absolute(convolution_k_fft2)
# plt.imshow(convolution_k_fft2_PS)
# plt.title("2D Power Spectrum, Gaussian Filter " + r'$\sigma$' + " = %.1f" % (sigma), fontsize=14)
# plt.colorbar()
# plt.show()

#####################################
# Find the power spectrum with rfft2
nx = np.fft.fftfreq(density.shape[0], d=1. / density.shape[0])[0:((density.shape[0] + 1) // 2)]
ny = np.fft.fftfreq(density.shape[1], d=1. / density.shape[1])[0:((density.shape[1] + 1) // 2)]
nx, ny = np.meshgrid(nx, ny)

density_PS = density_k[0:((density.shape[0] + 1) // 2), 0:((density.shape[1] + 1) // 2)]
density_PS = np.absolute(density_PS) ** 2
convolution_PS = convolution_k[0:((density.shape[0] + 1) // 2), 0:((density.shape[1] + 1) // 2)]
convolution_PS = np.absolute(convolution_PS) ** 2

density_mixed_PS = np.zeros(nx.shape[0] + nx.shape[1] - 1)
convolution_mixed_PS = np.zeros(nx.shape[0] + nx.shape[1] - 1)

n2_mixed = nx**2 + ny**2
n2_mixed = np.concatenate((n2_mixed[0, :], n2_mixed[1:, -1]), axis=None)

for i in range(nx.shape[0]):
    for j in range(nx.shape[1]):
        density_mixed_PS[i + j] = density_mixed_PS[i + j] + density_PS[i, j]
        convolution_mixed_PS[i + j] = convolution_mixed_PS[i + j] + convolution_PS[i, j]

# Plot the mixed nx^2 + ny^2 power spectrum
plot_range = 100
plt.plot(n2_mixed[1:plot_range], density_mixed_PS[1:plot_range], '.-b', label='Original')
plt.plot(n2_mixed[1:plot_range], convolution_mixed_PS[1:plot_range], '.-c', label='Gaussian ' + r'$\sigma$' + " = %.1f" % (sigma))
plt.xlabel(r'$n^2_x + n^2_y$', fontsize=12)
plt.title("Power Spectrum", fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.show()

# Plot the power spectrum in nx, and ny respectively
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title("Power Spectrum", fontsize=14)
ax.set_xlabel(r'$n_x$', fontsize=12)
ax.set_ylabel(r'$n_y$', fontsize=12)
ax.set_zlabel("ln|" + r'$F_n$' + "|" + r'$^2$', fontsize=12)
surf = ax.plot_surface(nx, ny, np.log(density_PS), cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title("Power Spectrum, Gaussian Filter " + r'$\sigma$' + " = %.1f" % (sigma), fontsize=14)
ax.set_xlabel(r'$n_x$', fontsize=12)
ax.set_ylabel(r'$n_y$', fontsize=12)
ax.set_zlabel("ln|" + r'$F_n$' + "|" + r'$^2$', fontsize=12)
surf = ax.plot_surface(nx, ny, np.log(convolution_PS), cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
