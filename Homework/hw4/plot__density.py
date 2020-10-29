import numpy as np
import matplotlib.pyplot as plt

filename = 'density.dat'
N = 1024
density = np.fromfile(filename, 'float32').reshape(N, N)

plt.imshow(density)
plt.colorbar()
plt.savefig('fig__density.png', bbox_inches='tight')
plt.show()
