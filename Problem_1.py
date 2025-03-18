import numpy as np
import matplotlib.pyplot as plt

#a
def visualize():
    # If your header is space-separated:
    data = np.loadtxt('mesh.dat', skiprows=1)

    x = data[:, 0]
    y = data[:, 1]

    # Create a scatter plot
    plt.scatter(x, y, c='blue', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud from mesh.dat')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    visualize()
