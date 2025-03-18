import numpy as np
import matplotlib.pyplot as plt

#Load point cloud data
data = np.loadtxt("data.dat")

#a
def graham_scan(points):
    points = sorted(points, key=lambda p: (p[0], p[1]))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])

#b
def jarvis_march(points):
    def leftmost_point(points):
        return min(points, key=lambda p: p[0])

    hull = []
    point_on_hull = leftmost_point(points)
    while True:
        hull.append(point_on_hull)
        endpoint = points[0]
        for p in points[1:]:
            if endpoint == point_on_hull or np.cross(endpoint - point_on_hull, p - point_on_hull) > 0:
                endpoint = p
        point_on_hull = endpoint
        if endpoint == hull[0]:
            break
    return np.array(hull)

#c
def quickhull(points):
    def find_hull(points, p1, p2):
        if not points:
            return []
        farthest = max(points, key=lambda p: np.cross(p2 - p1, p - p1))
        left_of_farthest = [p for p in points if np.cross(farthest - p1, p - p1) > 0]
        right_of_farthest = [p for p in points if np.cross(p2 - farthest, p - farthest) > 0]
        return find_hull(left_of_farthest, p1, farthest) + [farthest] + find_hull(right_of_farthest, farthest, p2)

    points = sorted(points, key=lambda p: p[0])
    leftmost, rightmost = points[0], points[-1]
    left_set = [p for p in points if np.cross(rightmost - leftmost, p - leftmost) > 0]
    right_set = [p for p in points if np.cross(leftmost - rightmost, p - rightmost) > 0]

    return np.array([leftmost] + find_hull(left_set, leftmost, rightmost) + [rightmost] + find_hull(right_set, rightmost, leftmost))

#d
def monotone_chain(points):
    points = sorted(points)
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])

#Convex Hulls
hull_graham = graham_scan(data)
hull_jarvis = jarvis_march(data)
hull_quickhull = quickhull(data)
hull_monotone = monotone_chain(data)

# Plot original points
plt.scatter(data[:, 0], data[:, 1], color='gray', alpha=0.5, label="Point Cloud")

# Plot convex hulls
plt.plot(hull_graham[:, 0], hull_graham[:, 1], 'r-', label="Graham Scan")
plt.plot(hull_jarvis[:, 0], hull_jarvis[:, 1], 'g-', label="Jarvis March")
plt.plot(hull_quickhull[:, 0], hull_quickhull[:, 1], 'b-', label="Quickhull")
plt.plot(hull_monotone[:, 0], hull_monotone[:, 1], 'm-', label="Monotone Chain")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Convex Hull Algorithms")
plt.savefig("convex_hull_plot.png")  # Save figure
plt.show()
