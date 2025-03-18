import numpy as np
import matplotlib.pyplot as plt
import timeit

#1a) 
def load_dat(file='mesh.dat', skiprows=1):
    pts = np.loadtxt(file, skiprows=skiprows)
    plt.figure()
    plt.scatter(pts[:,0], pts[:,1], c='blue')
    plt.title("mesh.dat")
    plt.show()
    return pts

#1b)
def graham_scan(points):
    def orientation(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    pts = sorted(points, key=lambda x:(x[0],x[1]))
    lower, upper = [], []
    for p in pts:
        while len(lower)>=2 and orientation(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper)>=2 and orientation(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(p)
    return np.array(lower[:-1]+upper[:-1])

def jarvis_march(points):
    pts = np.array(points)
    n = len(pts)
    if n<3: return pts
    hull = []
    l = np.argmin(pts[:,0])
    p = l
    while True:
        hull.append(p)
        q = (p+1)%n
        for i in range(n):
            if np.cross(pts[i]-pts[p], pts[q]-pts[p])>0:
                q = i
        p = q
        if p==l: break
    return pts[hull]

def quickhull(points):
    pts = np.unique(points, axis=0)
    if len(pts)<3: return pts
    def cross(o,a,b): return np.cross(a-o, b-o)
    def add_hull(sub, p1, p2):
        if not len(sub): return []
        dist = cross(p1,p2, sub)
        idx = np.argmax(dist)
        fp = sub[idx]
        left1 = sub[cross(p1,fp, sub)>0]
        left2 = sub[cross(fp,p2, sub)>0]
        return add_hull(left1, p1, fp)+[fp]+add_hull(left2, fp, p2)
    L = pts[np.argmin(pts[:,0])]
    R = pts[np.argmax(pts[:,0])]
    above = pts[cross(L,R,pts)>0]
    below = pts[cross(L,R,pts)<0]
    hull = [L]+add_hull(above,L,R)+[R]+add_hull(below,R,L)
    return np.array(hull)

def monotone_chain(points):
    pts = sorted(points, key=lambda x:(x[0],x[1]))
    def cross(o,a,b): return np.cross(a-o, b-o)
    lower, upper = [], []
    for p in pts:
        p = np.array(p)
        while len(lower)>=2 and cross(np.array(lower[-2]),np.array(lower[-1]),p)<=0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        p = np.array(p)
        while len(upper)>=2 and cross(np.array(upper[-2]),np.array(upper[-1]),p)<=0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1]+upper[:-1])

#1c) 
def showcase(points, hull_func, title):
    plt.figure()
    plt.title(title)
    plt.scatter(points[:,0], points[:,1], c='blue', label='Points')
    hull = hull_func(points)
    hull = np.vstack([hull, hull[0]])
    plt.plot(hull[:,0], hull[:,1], 'r', label='Hull')
    plt.legend()
    plt.show()

#2a) 
def point_cloud(n, min_val=0, max_val=1):
    return np.random.rand(n,2)*(max_val-min_val)+min_val

#2b) 
def run_time(n_list, min_val=0, max_val=1):
    gs, jm, qh, mc = [],[],[],[]
    for n in n_list:
        pts = point_cloud(n, min_val, max_val)
        gs.append(timeit.timeit(lambda: graham_scan(pts),    number=1))
        jm.append(timeit.timeit(lambda: jarvis_march(pts),   number=1))
        qh.append(timeit.timeit(lambda: quickhull(pts),      number=1))
        mc.append(timeit.timeit(lambda: monotone_chain(pts), number=1))
    return gs, jm, qh, mc

def plot_time_complexity(n_list, min_val=0, max_val=1):
    gs, jm, qh, mc = run_time(n_list, min_val, max_val)
    plt.figure()
    plt.plot(n_list, gs, label="Graham")
    plt.plot(n_list, jm, label="Jarvis")
    plt.plot(n_list, qh, label="Quickhull")
    plt.plot(n_list, mc, label="Monotone")
    plt.xlabel("Number of Points")
    plt.ylabel("Time (s)")
    plt.title(f"[{min_val},{max_val}]^2")
    plt.legend()
    plt.show()

#2c) 
def plot_time_histogram(n, runs, min_val=0, max_val=1):
    n_list = np.full(runs, n)
    gs, jm, qh, mc = run_time(n_list, min_val, max_val)
    for times, title in zip([gs,jm,qh,mc], ["Graham","Jarvis","Quickhull","Monotone"]):
        plt.figure()
        plt.hist(times, bins=20)
        plt.title(f"{title} histogram (n={n}, {runs} runs)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.show()

#2d) (Already covered by the histogram approach n=50, runs=100)

if __name__=="__main__":
    np.random.seed(1108)

    #1a) Load & plot .dat
    points_dat = load_dat('mesh.dat', skiprows=1)

    #1c) Showcase each hull
    showcase(points_dat, graham_scan,    "Graham Scan on mesh.dat")
    showcase(points_dat, jarvis_march,   "Jarvis March on mesh.dat")
    showcase(points_dat, quickhull,      "Quickhull on mesh.dat")
    showcase(points_dat, monotone_chain, "Monotone Chain on mesh.dat")

    #2b) Time complexity in [0,1]^2
    n_list = [10,50,100,200,400,800,1000]
    plot_time_complexity(n_list, 0, 1)

    #2b) Time complexity in [-5,5]^2
    plot_time_complexity(n_list, -5, 5)

    #2c) Histograms for n=50, repeated 100 runs, in [0,1]^2
    plot_time_histogram(50, 100, 0, 1)

    #Histograms for n=50, repeated 100 runs, in [-5,5]^2
    plot_time_histogram(50, 100, -5, 5)
