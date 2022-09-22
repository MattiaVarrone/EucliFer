from Triangulation import *

l = range(6)
combs = []
for i in range(6):
    for j in range(i+1, 6):
        for k in range(j+1, 6):
            combs.append([i, j, k])

comb_work = []
for comb in combs:
    m = Manifold(14, comb)
    works = True
    for i in range(len(m.adj)):
        U, def_triangles = circle_vertex(m.adj, m.sign, m.signc, i)
        trace_plaquette = np.trace(U) / 2
        if not np.isclose(trace_plaquette, np.cos(def_triangles * np.pi / 6)):
            works = False
            break
    if works:
        comb_work.append(comb)

print(comb_work)
print(len(comb_work))