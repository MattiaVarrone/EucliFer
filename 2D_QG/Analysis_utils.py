from Triangulation import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

eq_sweeps = 200
meas_sweeps = 2
n_measurements = 200


def make_profiles(beta, sizes, strategy=['gravity', 'ising'], eq_sweeps=eq_sweeps, n_measurements=n_measurements):
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


def finite_size_scaling(profile, sizes):
    prof_max, sigma_max = profiles_max(profile)
    fit = curve_fit(power_fit, sizes, prof_max, p0=[0.75, 1])
    (d, a), err = fit
    d_H = 1 / (1 - d)
    d_err = np.sqrt(err[0, 0])
    d_H_err = d_H ** 2 * d_err
    a_err = np.sqrt(err[1, 1])
    return d_H, d_H_err, a, a_err
