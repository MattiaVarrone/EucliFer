from Triangulation import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

eq_sweeps = 200
meas_sweeps = 2
n_measurements = 200


def dist_prof(adj,max_distance=30):
    '''Return array `profile` of size `max_distance` such that `profile[r]` is␣
    ↪the number
    of vertices that have distance r to a randomly chosen initial vertex.'''
    profile = np.zeros((max_distance),dtype=np.int32)
    neighbors = vertex_neighbors_list(adj)
    num_vertices = len(neighbors)
    start = rng.integers(num_vertices) # random starting vertex
    distance = np.full(num_vertices,-1,dtype=np.int32) # array tracking theknown distances (-1 is unknown)
    queue = deque([start])
    # use an exploration queue for the breadth-first search
    distance[start] = 0
    profile[0] = 1  # of course there is exactly 1 vertex at distance 0
    while queue:
        current = queue.pop()
        d = distance[current] + 1  # every unexplored neighbour will have this distance
        if d >= max_distance:
            break
        for nbr in neighbors[current]:
            if distance[nbr] == -1:  # this neighboring vertex has not been␣explored yet
                distance[nbr] = d
                profile[d] += 1
                queue.appendleft(nbr)
    # add it to the exploration queue
    return profile


def batch_estimate(data,observable,num_batches):
    batch_size = len(data)//num_batches
    values = [observable(data[i*batch_size:(i+1)*batch_size]) for i in range(num_batches)]
    return np.mean(values), np.std(values)/np.sqrt(num_batches-1)


def make_profiles(beta, sizes, strategy=['gravity', 'ising'], eq_sweeps=eq_sweeps, meas_sweeps=meas_sweeps, n_measurements=n_measurements):
    mean_profiles = []
    for size in sizes:
        m = Manifold(size)
        m.sweep(eq_sweeps, beta=beta, strategy=strategy)
        profiles = []
        for _ in range(n_measurements):
            m.sweep(meas_sweeps, beta=beta, strategy=strategy)
            profiles.append(dist_prof(m.adj, 15))
        mean_profiles.append([batch_estimate(data, np.mean, 20) for data in np.transpose(profiles)])
        print(size)
    return np.array(mean_profiles)


def scale_profile(profiles, sizes, d):
    xs, ys = [], []
    for i, profile in enumerate(profiles):
        rvals = np.arange(len(profile))
        x = rvals / sizes[i] ** d
        y = profile / sizes[i] ** (1 - d)
        xs.append(x), ys.append(y)
        # plt.plot(x, y)
    return np.array(xs), np.array(ys)


def plot_profiles(mean_profiles, sizes):
    for profile in mean_profiles:
        plt.plot([y[0] for y in profile])
        plt.fill_between(range(len(profile)), [y[0] - y[1] for y in profile], [y[0] + y[1] for y in profile], alpha=0.2)
    plt.legend(sizes, title="N")
    plt.xlabel("r")
    plt.ylabel(r"$\mathbb{E}[\rho_T(r)]$")
    plt.title("Distance profile")
    plt.show()


def overlay_profiles(profiles, ranges, sizes):
    for profile, range in zip(profiles, ranges):
        plt.plot(range, [y[0] for y in profile])
        # plt.fill_between(range, range(len(profile)), [y[0]-y[1] for y in profile], [y[0]+y[1] for y in profile], alpha=0.2)
    plt.legend(sizes, title="N")
    plt.xlabel("r")
    plt.ylabel(r"$\mathbb{E}[\rho_T(r)]$")
    plt.title("Scaled distance profile")
    plt.show()


def profiles_max(profiles):
    profiles_max = np.max(profiles, axis=1)
    return profiles_max[:, 0], profiles_max[:, 1]


def power_fit(N, d, a):
    return a * N ** d


def lin_fit(x, d, b):
    return b + d*x


def finite_size_scaling(profile, sizes):
    prof_max, sigma_max = profiles_max(profile)
    fit = curve_fit(power_fit, sizes, prof_max, p0=[0.75, 1])
    (d, a), err = fit
    d_H = 1 / (1 - d)
    d_err = np.sqrt(err[0, 0])
    d_H_err = d_H ** 2 * d_err
    a_err = np.sqrt(err[1, 1])
    return d_H, d_H_err, a, a_err
