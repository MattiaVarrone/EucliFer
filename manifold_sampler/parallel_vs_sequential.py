from Analysis_utils import *

beta = 0.5
sizes = [(int(i) // 2) * 2 for i in np.geomspace(10, 24, 4)]

prof = make_profiles(sizes, beta, meas_sweeps=2, strategy=['gravity', 'spinor'])
def profile_maker(size):
    return make_profile(size, 1, ['gravity', 'spinor'], eq_sweeps, meas_sweeps, n_measurements)
prof2 = make_profiles_mp(sizes, profile_maker)