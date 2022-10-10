import os
from Analysis_utils import *
from scipy.interpolate import CubicSpline

# params for the lattice-matter system and sampling
sizes = [(int(i) // 4) * 4 + 2 for i in np.geomspace(30, 100, 8)]
beta = 0.63
matter = 'spinor_free'
strategy = ['gravity', matter]
eq_sweeps, meas_sweeps, n_measurements = 200, 4, 200

# getting the array index of the hpc batch
array_id = os.getenv('SLURM_ARRAY_TASK_ID')
if array_id is not None:
    array_id = int(array_id)


# create a distance profile for different lattice sizes
def profile_maker(size):
    return make_profile(size, beta, strategy, eq_sweeps, meas_sweeps, n_measurements)


if __name__ == '__main__':

    # the process of creating and updating different lattices is parallelised.
    pool = mp.Pool(mp.cpu_count())
    with timebudget("Time to make profiles:"):
        prof = np.array(pool.map(profile_maker, sizes))

    # create plots
    fig, ax = plt.subplots(1, 2)

    # fit a cubic spline through the distance profiles
    fit_prof = []
    prof_0 = prof[..., 0]  # neglect errors
    rs = np.arange(0.5, 10, 0.05)  # range for spline fit

    for profile in prof_0:
        dist_range = np.arange(len(profile))
        cs = CubicSpline(dist_range, profile)
        fit_prof.append(cs(rs))

        # plot distance profiles
        ax[0].plot(dist_range, profile, 'x')
        ax[0].plot(rs, cs(rs))

    np.savez('data/profiles_spinor_1', full_prof=prof, fit_prof=np.array(fit_prof), prof_0=prof_0, sizes=sizes,
             beta=beta)

    # make fit of finite size scaling in log-log space
    y = np.log(np.max(fit_prof, axis=1))
    x = np.log(sizes)
    fit = curve_fit(lin_fit, x, y, p0=[0.75, 0])
    (d, a), err = fit
    d_H = 1 / (1 - d)
    d_err = np.sqrt(err[0, 0])
    d_H_err = d_H ** 2 * d_err
    a_err = np.sqrt(err[1, 1])

    # make fit without spline
    d_H_ws, d_H_err_ws, a_ws, a_err_ws = finite_size_scaling(prof, sizes)

    ##### plot settings #####
    ax[1].plot(x, lin_fit(x, d, a), 'g')
    ax[1].plot(x, lin_fit(x, 1 - 1 / d_H_ws, a_ws))
    ax[1].plot(x, y, 'gx')

    ax[0].set_title('Distance Profiles'), ax[1].set_title('$Log(r_{max})\; vs\; Log(N)$')
    ax[0].legend(["N = " + str(N) for N in sizes]), ax[1].legend(["Spline", "No spline"])
    fig.suptitle(f'{matter = }, {beta = }')

    ID = "" if array_id is None else str(array_id)
    fig.set_size_inches(8, 5)
    plt.savefig("data/pics/dist_" + matter + "_" + ID + ".png")
    plt.show()

    ##### print results and info #####
    print('Number of cores: ' + str(mp.cpu_count()))
    print(f'{array_id = }')

    print("\nwith spline fit:")
    print(f'{d_H = }' + '+/-' + f'{d_H_err = }')
    print(f'{a = }' + '+/-' + f'{a_err = }')

    print("\nwithout spline fit:")
    print(f'{d_H_ws = }' + '+/-' + f'{d_H_err_ws = }')
    print(f'{a_ws = }' + '+/-' + f'{a_err_ws = }')
