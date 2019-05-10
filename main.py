import numpy as np
import matplotlib.pyplot as plt
from RBF import RBF
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--functions', help='number of radial functions', default=8)
parser.add_argument('-s', '--sigma', help='sigma for radial functions', default=1.0)
parser.add_argument('-o', '--save', help='boolean for saving the figure', default=True)
args = parser.parse_args()

# real function
x = np.linspace(-2, 2, 100)
y = np.power(x, 3)

model = RBF(hidden_shape=int(args.functions), sigma=float(args.sigma))

model.fit(x, y)
predicted_y = model.predict(x)

plt.plot(x, y, 'b-', label='real function in [-2,2]')
plt.plot(x, predicted_y, 'r-', label='fit')
plt.legend(loc='upper left')
plt.title('Function Approximation with RBF')
if args.save and bool(args.save):
    plt.savefig('RBF-Trained-With-' + str(args.functions) + '-Radials.png')
plt.show()
