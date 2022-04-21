import yt 
import numpy as np 
import animation 

import yt_idv


obj = animation.loadRun("3d_runs/hybrid_scheme_15.npz")
config, psi, density, phase = obj
data = dict(density = (density, "g/cm**3"), phase = (phase, "g/cm**3"))

bbox = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
ds = yt.load_uniform_grid(data, density.shape, length_unit="Mpc", bbox=bbox, nprocs=64)

rc = yt_idv.render_context()
rc.add_scene(ds, "density")
rc.run()