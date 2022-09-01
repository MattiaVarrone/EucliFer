import os
from Analysis_utils import *
from scipy.interpolate import CubicSpline

sizes = [(int(i) // 2) * 2 for i in np.geomspace(30, 70, 4)]
beta = 0.5
matter = 'ising'
strategy = ['gravity', matter]
eq_sweeps, meas_sweeps, n_measurements = 200, 4, 200


def profile_maker(size):
    return make_profile(size, beta, strategy, eq_sweeps, meas_sweeps, n_measurements)


array_id = os.getenv('SLURM_ARRAY_TASK_ID')
print(array_id)

prof = make_profiles_mp(sizes, profile_maker)

prof_0 = prof[..., 0]
rs0 = np.arange
rs = np.arange(0.5, 10, 0.05)
fit_prof = []

fig, ax = plt.subplots(1, 2)

for profile in prof_0:
    dist_range = range(len(profile))
    cs = CubicSpline(dist_range, profile)
    fit_prof.append(cs(rs))
    ax[0].plot(dist_range, profile, 'x')

for p in fit_prof:
    ax[0].plot(rs, p)

y = np.log(np.max(fit_prof, axis=1))
x = np.log(sizes)
ax[1].plot(x, y, 'gx')

ax[0].set_title('Distance Profiles'), ax[1].set_title('$Log(r_{max})\; vs\; Log(N)$')
ax[0].legend(["N = " + str(N) for N in sizes])

np.savez('data/profiles_spinor_1', full_prof=prof, fit_prof=np.array(fit_prof), prof_0=prof_0, sizes=sizes, beta=beta)

fit = curve_fit(lin_fit, x, y, p0=[0.75, 0])
(d, a), err = fit
ax[1].plot(x, lin_fit(x, d, a))
d_H = 1 / (1 - d)
d_err = np.sqrt(err[0, 0])
d_H_err = d_H ** 2 * d_err
a_err = np.sqrt(err[1, 1])

print('Number of cores: ' + str(mp.cpu_count()))

print("\nwith spline fit:")
print(f'{d_H = }' + '+/-' + f'{d_H_err = }')
print(f'{a = }' + '+/-' + f'{a_err = }')

d_H, d_H_err, a, a_err = finite_size_scaling(prof, sizes)
ax[1].plot(x, lin_fit(x, 1 - 1 / d_H, a))
print("\nwithout spline fit:")
print(f'{d_H = }' + '+/-' + f'{d_H_err = }')
print(f'{a = }' + '+/-' + f'{a_err = }')

ID = "" if array_id is None else str(array_id)
plt.savefig("data/pics/dist_" + matter + "_" + ID + ".png")
plt.show()
