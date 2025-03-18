import numpy as np
import matplotlib.pyplot as plt
import time


#1a) 

def load_and_visualize_dat(dat_file='mesh.dat', skip_header=True):
    if skip_header:
        points = np.loadtxt(dat_file, skiprows=1)
    else:
        points = np.loadtxt(dat_file)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], color='blue')
    plt.title("1a) Point Cloud from '{}'".format(dat_file))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
    return points

#1b)

def graham_scan(points):
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    pts = sorted(points, key=lambda x: (x[0], x[1]))
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and orient(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and orient(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])

def jarvis_march(points):
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    pts = np.array(points)
    n = len(pts)
    if n < 3:
        return pts

    hull = []
    leftmost = np.argmin(pts[:, 0])
    p = leftmost
    while True:
        hull.append(p)
        q = (p + 1) % n
        for i in range(n):
            o = orient(pts[p], pts[i], pts[q])
            # If more counterclockwise OR collinear but farther
            if o > 0 or (o == 0 and np.sum((pts[p] - pts[i])**2) > np.sum((pts[p] - pts[q])**2)):
                q = i
        p = q
        if p == leftmost:
            break
    return pts[hull]

def quickhull(points):
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    pts = np.unique(points, axis=0)  # remove duplicates
    if len(pts) < 3:
        return pts

    L = pts[np.argmin(pts[:, 0])]
    R = pts[np.argmax(pts[:, 0])]
    hull = [L, R]

    def split(S, A, B):
        if not len(S):
            return
        distmax = -1
        C = None
        for p in S:
            d = abs(orient(A, B, p))
            if d > distmax:
                distmax = d
                C = p
        hull.insert(hull.index(B), C)
        S1 = [p for p in S if orient(A, C, p) > 0]
        S2 = [p for p in S if orient(C, B, p) > 0]
        split(S1, A, C)
        split(S2, C, B)

    left_set  = [p for p in pts if orient(L, R, p) > 0]
    right_set = [p for p in pts if orient(R, L, p) > 0]
    split(left_set, L, R)
    split(right_set, R, L)

    hull_arr = np.array(hull)
    c = np.mean(hull_arr, axis=0)
    angles = np.arctan2(hull_arr[:,1]-c[1], hull_arr[:,0]-c[0])
    hull_arr = hull_arr[np.argsort(angles)]
    return hull_arr

def monotone_chain(points):
    """Compute the convex hull using the Monotone Chain algorithm."""
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    pts = sorted(points, key=lambda x: (x[0], x[1]))
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and orient(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and orient(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])

#1c)

def plot_all_hulls_in_one_figure(points):
    hulls = [
        (graham_scan(points),    'r', 'Graham Scan'),
        (jarvis_march(points),   'g', 'Jarvis March'),
        (quickhull(points),      'b', 'Quickhull'),
        (monotone_chain(points), 'm', 'Monotone Chain')
    ]
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='k', label='Points')
    for hull, color, name in hulls:
        hull = np.vstack([hull, hull[0]])  # close the loop
        plt.plot(hull[:, 0], hull[:, 1], color, label=name)
    plt.title("1c) Convex Hulls (All Methods)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()

#2a)
def generate_uniform_points(n):
    return np.random.rand(n, 2)

#2b)
def measure_and_plot_time_complexity_uniform():
    ns = [10, 50, 100, 200, 400, 800, 1000]
    times_graham = []
    times_jarvis = []
    times_quick  = []
    times_mono   = []
    for n in ns:
        points = generate_uniform_points(n)
        # Graham
        start = time.time()
        graham_scan(points)
        times_graham.append(time.time() - start)
        # Jarvis
        start = time.time()
        jarvis_march(points)
        times_jarvis.append(time.time() - start)
        # Quickhull
        start = time.time()
        quickhull(points)
        times_quick.append(time.time() - start)
        # Monotone
        start = time.time()
        monotone_chain(points)
        times_mono.append(time.time() - start)

    plt.figure()
    plt.plot(ns, times_graham, 'r-o', label='Graham Scan')
    plt.plot(ns, times_jarvis, 'g-o', label='Jarvis March')
    plt.plot(ns, times_quick,  'b-o', label='Quickhull')
    plt.plot(ns, times_mono,   'm-o', label='Monotone Chain')
    plt.title("2b) Time Complexity in [0,1]^2 (Uniform)")
    plt.xlabel("Number of Points (n)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.legend()
    plt.show()

#2(d)
def part_d_runtime_histograms(n=50, n_runs=100, distribution='uniform', seed=42):
    np.random.seed(seed)
    times_graham = []
    times_jarvis = []
    times_quick  = []
    times_mono   = []

    for _ in range(n_runs):
        # Generate points
        if distribution == 'uniform':
            # Uniform in [-5, 5]
            points = np.random.uniform(-5, 5, size=(n, 2))
        else:
            # Gaussian with mean=0, variance=1
            points = np.random.randn(n, 2)

        # Graham
        start = time.time()
        graham_scan(points)
        times_graham.append(time.time() - start)

        # Jarvis
        start = time.time()
        jarvis_march(points)
        times_jarvis.append(time.time() - start)

        # Quickhull
        start = time.time()
        quickhull(points)
        times_quick.append(time.time() - start)

        # Monotone
        start = time.time()
        monotone_chain(points)
        times_mono.append(time.time() - start)

    #4 histograms in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0,0].hist(times_graham, bins=10, color='r')
    axs[0,0].set_title("Graham Scan Times")
    axs[0,1].hist(times_jarvis, bins=10, color='g')
    axs[0,1].set_title("Jarvis March Times")
    axs[1,0].hist(times_quick, bins=10, color='b')
    axs[1,0].set_title("Quickhull Times")
    axs[1,1].hist(times_mono, bins=10, color='m')
    axs[1,1].set_title("Monotone Chain Times")

    plt.suptitle(f"2(d) Runtime Distributions (n={n}, {distribution}, {n_runs} runs)")
    plt.tight_layout()
    plt.show()

    stats = {
        'Graham': (min(times_graham), max(times_graham), np.mean(times_graham)),
        'Jarvis': (min(times_jarvis), max(times_jarvis), np.mean(times_jarvis)),
        'Quick':  (min(times_quick),  max(times_quick),  np.mean(times_quick)),
        'Mono':   (min(times_mono),   max(times_mono),   np.mean(times_mono))
    }

    with open("part_d_conclusion.txt", "w") as f:
        f.write(f"Part (d) Conclusion - distribution='{distribution}', n={n}, runs={n_runs}\n")
        f.write("Format: (best_time, worst_time, mean_time)\n")
        for alg, (best_, worst_, mean_) in stats.items():
            f.write(f"{alg}: best={best_:.6f}, worst={worst_:.6f}, mean={mean_:.6f}\n")
        f.write("\nFrom the histograms, you can discuss which algorithm typically has\n")
        f.write("the fastest or slowest times, and whether the distribution of times\n")
        f.write("appears roughly normal, lognormal, etc.\n")

#final
if __name__ == "__main__":
    #1a)
    points_dat = load_and_visualize_dat('mesh.dat', skip_header=True)

    #1c)
    plot_all_hulls_in_one_figure(points_dat)

    #2b)
    np.random.seed(42)
    measure_and_plot_time_complexity_uniform()

    #2(d)
    part_d_runtime_histograms(n=50, n_runs=100, distribution='uniform', seed=42)

    # 2(d) (optional)
    part_d_runtime_histograms(n=50, n_runs=100, distribution='gaussian', seed=42)
