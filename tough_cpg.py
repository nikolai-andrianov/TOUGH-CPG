"""
A 3D CO2 injection case in a 3D corner-point geometry (CPG) using the TOUGHREACT-ECO2N model.

The code below is intended to demonstrate a workflow for using corner-point grids with TOUGH/TOUGHREACT.

Copyright 2023 Nikolai Andrianov, nia@geus.dk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from t2data import *
from t2listing import *
import xtgeo
import pyvista as pv
from shapely.geometry import Polygon
import subprocess
import os, shutil
from glob import glob
import math
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from rampup_timesteps import *
import pandas as pd
import time
import itertools
import sys

np.seterr(all='raise')

# Input corner-point geometry, the TOUGH* executable, and the run folder
grdecl = 'faulted_5x5x2.GRDECL'
# grdecl = 'faulted_10x10x5.GRDECL'
# grdecl = 'faulted_15x15x7.GRDECL'
#grdecl = 'faulted_30x30x10.GRDECL'

exe = 'treactv332omp_eco2n_pc.exe'
folder = grdecl.split('.')[0]

# Setting up the CPG geometry using pyTOUGH is time-consuming for large grid sizes
pytough_grid = True
#pytough_grid = False

print('\nSetting up pyTOUGH parameters..\n')
start_time = time.time()

# Shorthands
darcy = 9.869232667160130e-13
poise = 0.1
barsa = 1e5
day = 86400.
year = 31556952.

# -----------------------------------------------------------------------------
# Create an empty input data structure
dat = t2data()
dat.title = 'CO2 injection case in a corner-point geometry using the ECO2N model'
dat.start = True

# -----------------------------------------------------------------------------
# ROCKS section

# Uniform matrix porosity and permeability
phi = 0.25
K = 250*1e-3*darcy
Kx, Ky, Kz = K, K, K

# Pore compressibility
cr = 0

# Create the rock with custom relative permeabilities and cap pressures (nad=2)
rock = rocktype(name='SAND', nad=2, density=2600,
                porosity=phi, permeability=[Kx, Ky, Kz],
                conductivity=2.51, specific_heat=920)

# Set additional rock parameters
rock.compressibility = cr

# # Type 10 = modified Brooks-Corey
# # Parameters: [Slr, Sgr, switch] = [0, 0, 0], see TOUGH3 User Guide p. 138
# rock.relative_permeability = {'type': 10,
#                               'parameters': [0, 0, 0]}
#
# # Type 10 = modified Brooks-Corey
# # Parameters: [lambda, Pe, eps, Slr] = [-2, 0, 0, 0], see TOUGH3 UG p. 150
# rock.capillarity = {'type': 7,
#                     'parameters': [-2, 0, 0, 0]}

# The parameters for the van Genuchten relative permeabilities
# Yield 1 at irreducible saturations, not consistent with the Corey rel perm in MRST
m_rp = 1.1
slr_rp = 0.27
sls_rp = 1
sgr_rp = 1e-5  #0.05

# The parameters for the van Genuchten capillary pressure
m_cp = 0.457
slr_cp = slr_rp
p0_cp = 5e-5
pmax_cp = 1e7
sls_cp = sls_rp

plot_kr = False
if plot_kr:
    sw = np.linspace(slr_rp, sls_rp)
    ss = (sw - slr_rp) / (sls_rp - slr_rp)
    krl = np.sqrt(ss) * pow(1 - pow(1 - pow(ss, 1/m_rp), m_rp), 2)
    sg = (sw - slr_rp) / (1 - slr_rp - sgr_rp)
    krg = pow(1 - sg, 2) * (1 - pow(sg, 2))

    fig = plt.figure()
    plt.plot(sw, krl)
    plt.plot(sw, krg)
    plt.show(block=False)

    pc = - 1/p0_cp * pow(pow(ss, -1/m_cp) - 1, 1 - m_cp)
    pc = np.minimum(pmax_cp, - pc)
    fig = plt.figure()
    plt.plot(sw, pc)
    plt.show(block=False)


# Type 7 = van Genuchten
# Parameters: [m, slr, sls, sgr], see TOUGH3 User Guide p. 144
rock.relative_permeability = {'type': 7,
                              'parameters': [m_rp, slr_rp, sls_rp, sgr_rp]}

# Type 7 = van Genuchten
# Parameters: [m, slr, 1/P0, pmax, sls], see TOUGH3 UG p. 156
rock.capillarity = {'type': 7,
                    'parameters': [m_cp, slr_cp, p0_cp, pmax_cp, sls_cp]}


# -----------------------------------------------------------------------------
#  MULTI section
dat.multi = {'num_components': 3,
             'num_equations': 3,
             'num_phases': 3,
             'num_secondary_parameters': 6}

# -----------------------------------------------------------------------------
#  PARAM section

# Run options
tend = 2 * year    # Final simulation time
#tend = 33750 * 10
dt = year / 12      # Timestep
#dt = 33750
#nout = 10            # Save results every nout-th time step

# Small time steps
tend = year / 12
dt = tend / 10

# Initial pressure and temperature
# Initial pressure is the mean of the MRST pressure
p0 = 5.*barsa
t0 = 75.     # degC

# Initial conditions [pressure_pa, massfrac_NaCl_liq, massfrac_CO2_Liq, temp_c]
# See TOUGH2-ECO2N_V2_Users_Guide p. 10
incons = [p0, 0, 0, t0]

dat.parameter.update(
    {'print_level': 4,
     'max_timesteps': 9999,
     'print_interval': 9999,
     'tstop': tend,
     'const_timestep': -1.0,
     'timestep': [dt],
     'max_timestep': 0.1*dt,
     'print_block': 'A1 48',
     'gravity': 9.81,
     'relative_error': 0.001,
     'absolute_error': 1.0,
     'default_incons': incons})

# Set options:
dat.parameter['option'][12] = 1 # interpolation procedure for time dependent sink/source data
dat.parameter['option'][16] = 4 # automatic time step control.
dat.parameter['option'][21] = 5 # The linear equation solver

# -----------------------------------------------------------------------------
#  SELEC section

# Select the solubility option by setting IE(16)=2, see TOUGH2-ECO2N_V2_Users_Guide
# Apparently IE(16) = 2 chooses CO2, = 3 is for methane.
dat.selection = {'integer': [None]*16, 'float': [None]*8}
dat.selection['integer'][0] = 1
dat.selection['integer'][9:15] = [0] * 6
dat.selection['integer'][15] = 2
# parameters for functional dependence of permeability on solid saturation,
# see TOUGH2-ECO2N_V2_Users_Guide p. 44
dat.selection['float'][0] = 0.8
dat.selection['float'][1] = 0.8

# -----------------------------------------------------------------------------
# REACT section

# Parameter choices for reactive transport simulation, see TOUGHREACT_V332_RefManual p. 19
# Not implemented in pyTOUGH

# -----------------------------------------------------------------------------
# TIMES section

# Printout at specified times
times = rampup_timesteps(tend, dt, n=20)
# Need to add time 0 for the proper GENER setup
times = np.concatenate(([0], times))
output_times = list(np.cumsum(times))
dat.output_times['time'] = output_times
dat.output_times['num_times_specified'] = len(dat.output_times['time'])

end_time = time.time()
dt_setup = end_time - start_time
print("Setting pyTOUGH parameters: %s sec" % dt_setup)

# -----------------------------------------------------------------------------
# Mesh generation

print('\nReading corner-point mesh from ' + grdecl + '\n')
start_time = time.time()

# Read the CPG geometry
#grd = xtgeo.grid_from_file('cartesian.GRDECL')
grd = xtgeo.grid_from_file(grdecl)
dim, crn, inactind = grd.get_vtk_geometries()
#gstuff = grd.get_geometrics(return_dict=True)
cpg = pv.ExplicitStructuredGrid(dim, crn)
cpg = cpg.compute_connectivity()
cpg = cpg.compute_connections()
cpg = cpg.compute_cell_sizes(length=False, area=False, volume=True)
cpg.flip_z(inplace=True)

#cpg.flip_normal([1, 0, 0], inplace=True)

n_nodes = cpg.dimensions
n_cells = [n - 1 for n in n_nodes]
dims = 'x'.join([str(d) for d in n_cells])
print( ' Read ' + dims + '=' + str(cpg.n_cells) + ' cells\n')

# Hiding cells does not remove them from the grid
#cpg.hide_cells(inactind, inplace=True)
# cpg.plot(show_edges=True)

# cpg_z = cpg.scale([1, 1, 10], inplace=False)
# cpg_z.plot(show_edges=True)
# cpg_z.save('cpg_z.vtk')

# Get the cell centers
centers = cpg.cell_centers()
#edge_centers = cpg.extract_all_edges().cell_centers().points

# # Plotting
# pl = pv.Plotter()
# cpg_z = cpg.scale([1, 1, 1], inplace=False)
# pl.add_mesh(cpg_z, show_edges=True, line_width=1, opacity=0.85)
# #pl.add_mesh(centers, color="r", point_size=8.0, render_points_as_spheres=True)
# pl.show()
#sys.exit(0)

end_time = time.time()
dt_cpg = end_time - start_time
print("Reading CPG: %s sec" % dt_cpg)

# Create a dummy Cartesian geometry with the same dimension as CPG, in order to set up the cell and connection lists.
# cpg.dimensions refer to nodes, mulgrid geometry requires cells' dimensions

print('\nCreating MULgrid from CPG\n')
start_time = time.time()

# n_nodes = cpg.dimensions
# n_cells = [n - 1 for n in n_nodes]

# Use convention=2 to get potentially up to 999 layers (see mulgrids.py line 565)
dd = [[999] * n for n in n_cells]
geo = mulgrid().rectangular(dd[0], dd[1], dd[2])  # , convention=2

# Set the grid data fields
dat.grid = t2grid()
dat.grid.add_rocktype(rock)
dat.grid.add_blocks(geo)
dat.grid.add_connections(geo)

end_time = time.time()
dt_mulgrid = end_time - start_time
print("Creating MULgrid: %s sec" % dt_mulgrid)

# Do not consider removing inactive cells for the moment
assert dat.grid.num_blocks == cpg.n_cells

# Map the cell indices to block names in TOUGH notation
block_name = [b for b in dat.grid.block]

# Setting up the CPG geometry using pyTOUGH is time-consuming for large grid sizes
if pytough_grid:
    print('\nSetting up the grid in pyTOUGH..\n')
    start_time = time.time()

    # Prepare fast-access lists
    point_id = []
    for c in cpg.cell:
        point_id.append(c.point_ids)

    # Print message when execution time is > check_min minutes
    check_min = 1

    # First fix the block centers and volumes to match the CPG geometry
    for i, (b, c) in enumerate(zip(dat.grid.block, cpg.cell)):
        dat.grid.block[b].centre = np.mean(c.points, axis=0)
        dat.grid.block[b].volume = cpg.cell_data['Volume'][i]

    # Next, fix the connections data
    dat.grid.connection_old = dat.grid.connection
    dat.grid.connectionlist_old = dat.grid.connectionlist
    dat.grid.connection = {}
    dat.grid.connectionlist = []
    for i, (b, c) in enumerate(zip(dat.grid.block, cpg.cell)):

        # Clear the sets as defined from the dummy Cartesian grid
        dat.grid.block[b].connection_name.clear()
        dat.grid.block[b].neighbour_name.clear()


        for j in cpg.neighbors(i):

            # The connection name
            cname = (b, block_name[j])

            # Identify the face, connecting the neighbors by comparing the joint points for the 2 neighboring cells
            pi = c.point_ids
            #pj = cpg.cell[j].point_ids
            pj = point_id[j]
            pc = set(pi) & set(pj)
            for f in c.faces:

                if set(f.point_ids) == pc:
                    # Check if this connection has not been added before
                    conn_added = False

                    for conn in dat.grid.connection.keys():
                        if (conn[0] == cname[0] and conn[1] == cname[1]) or (conn[0] == cname[1] and conn[1] == cname[0]):
                            conn_added = True
                            break

                    if not conn_added:
                        # Neighboring blocks
                        blocks = [dat.grid.block[b], dat.grid.block[block_name[j]]]
                        # Leave the default permeability direction = 1
                        direction = 1
                        # Distances between the neighboring blocks midpoints to the face8
                        face_center = np.mean(f.points, axis=0)
                        d0 = np.linalg.norm(blocks[0].centre - face_center)
                        d1 = np.linalg.norm(blocks[1].centre - face_center)
                        dist = [d0, d1]
                        # Face area
                        area = f.cast_to_unstructured_grid().area
                        # All block centre's have been updated in a loop above; 0-th is the current block, 1st is the neighbor
                        d = blocks[1].centre - blocks[0].centre
                        # Cosine of the angle between the gravitational acceleration vector and the line between the two blocks
                        try:
                            # Quick fix
                            if np.linalg.norm(d) < 1e-8:
                                dircos = 0
                            else:
                                dircos = np.dot(d, geo.tilt_vector) / np.linalg.norm(d)
                        except:
                            print('Error in dircos')

                        # Add connection to dat.grid.connectionlist, dat.grid.connection, and to block.connection_name
                        # dat.grid.add_connection(t2connection(blocks, direction, dist, area, dircos))
                        newconnection = t2connection(blocks, direction, dist, area, dircos)

                        # The original dat.grid.add_connection(tc) is too slow because of the call to
                        # self.connection[conname] = newconnection in t2grids.py line 317
                        #dat.grid.add_connection(newconnection)

                        # Contents of add_connection() from t2grids.py
                        conname = tuple([blk.name for blk in newconnection.block])
                        if conname in dat.grid.connection:
                            i = dat.grid.connectionlist.index(dat.grid.connection[conname])
                            dat.grid.connectionlist[i] = newconnection
                        else:
                            dat.grid.connectionlist.append(newconnection)
                        #dat.grid.connection[conname] = newconnection
                        for block in newconnection.block: block.connection_name.add(conname)

                        end_time = time.time()
                        dt_cpgrid = end_time - start_time
                        if dt_cpgrid / 60 > check_min:
                            print("   %s min.." % check_min)
                            check_min += 1

    end_time = time.time()
    dt_cpgrid = end_time - start_time
    print("Setting up the grid in pyTOUGH: %s sec" % dt_cpgrid)

# -----------------------------------------------------------------------------
# GENER section

# Show the perforated cell in the middle of the bottom layer (-1 to account for 0-based numbering)
iperf = round(n_cells[0] / 2)
jperf = round(n_cells[1] / 2)
kperf = n_cells[2] - 1
perf_id = cpg.cell_id((iperf, jperf, kperf))

# Does not work well..
plot_perf = True
if plot_perf:
    perf = np.zeros(cpg.number_of_cells)
    perf[perf_id] = 1
    cpg.cell_data['Perf_cells'] = perf
    cpg = cpg.scale([1, 1, 10], inplace=True)
    pl = pv.Plotter()
    #pl.add_mesh(cpg.outline(), color="k")
    #cpg_z.plot(scalars='Perf_cells', show_edges=True)
    pl.add_mesh(cpg, show_edges=True, line_width=1, opacity=0.5)
    #pl.add_mesh(perf, color="r", point_size=8.0, render_points_as_spheres=True)
    pl.add_mesh(cpg, scalars='Perf_cells')
    
    pl.show_grid()
    pl.show()

    # Plot well as a vertical line, ending at the performation
    #pl = pv.Plotter()
    #well = np.array([[iperf, jperf, 0], [iperf, jperf, kperf]])
    #pl.add_lines(well, color='yellow', width=3)
    #pl.add_mesh(cpg, show_edges=True, line_width=1, opacity=0.5, render_lines_as_tubes=True)
    #pl.show()


# CO2 injection of 1.5 MTa
q = 1.5 * 1e6 * 1e3 / year

# A mass rate (kg/sec) source in the perforated block, injecting component 3 (COM3)
# with given specific enthalpy in ltab time steps.
# See see TOUGH3 UG p. 85
# The generator name has be in the format "%3s%2d"

gener = t2generator(name='INJ 1', block=block_name[perf_id], type='COM3',
                    ltab=2, itab='h', gx=None, ex=.538E+05, hg=None, fg=None,
                    time=[0, times[-2]],
                    rate=[q, q],
                    enthalpy=[.538E+05, .538E+05])

dat.add_generator(gener)

# -----------------------------------------------------------------------------
# Run TOUGHREACT in a clean subfolder of the current folder
try:
    if os.path.isdir(folder):
        print('\nDeleting everything in ' + folder + '\n')
        shutil.rmtree(folder)
    os.mkdir(folder)
except Exception as e:
    print('Failed to create a clean %s. Reason: %s' % (folder, e))

print('\nRunning ' + exe + ' in ' + folder + '\n')
start_time = time.time()

try:
    # Copy CO2TAB to the run folder
    shutil.copyfile('CO2TAB', folder + '/CO2TAB')
    # Save the input file and run the simulation
    dat.write(folder + '/' + 'flow.inp')
    shutil.copyfile(exe, folder + '/' + exe)
    subprocess.run(exe, cwd=folder)
except Exception as e:
    print('Failed to run. Reason: %s' % e)

end_time = time.time()
dt_exe = end_time - start_time
print("Simulator run time: %s sec" % dt_exe)

# -----------------------------------------------------------------------------
# Apparently TOUGHREACT can output values < 1e-99 which are incorrectly
# represented in TEC files with omitted exponential notation.
# Fixing this..

print('\nFixing run results\n')
start_time = time.time()

nfix = 0
with open(folder + '/flowdata.tec', 'rt') as f:
    lines = f.readlines()

    for m, l in enumerate(lines):
        if l[:2] == '  ':
            par = ' '.join(l.split())  # Replace multiple spaces with one
            token = par.split(' ')
            for n, t in enumerate(token):
                if not 'E' in t:
                    nfix += 1
                    if '-' in t: token[n] = '{:.6E}'.format(0)
                    if '+' in t: token[n] = '{:.6E}'.format(1e99)
            if nfix > 0:
                lines[m] = '  ' + '  '.join(token) + '\n'

# Eventually replace the original TEC with the parsed one
if nfix > 0:
    print(f'\n{nfix} erroneus values in scientifc notation found in ' +
          folder + '/flowdata.tec')
    shutil.copyfile(folder + '/flowdata.tec', folder + '/flowdata_nonfix.tec')
    print('Copied the original file to ' + folder + '/flowdata_nonfix.tec')
    with open(folder + '/flowdata.tec', 'w') as f:
        f.writelines(lines)
    print('Saved the fixed file to ' + folder + '/flowdata.tec')

end_time = time.time()
dt_fix = end_time - start_time
print("Fixing run results: %s sec" % dt_fix)

# Get the ToughReact simulation results
print('\nSaving VTK results in ' + folder + ' ...')
start_time = time.time()

lst = toughreact_tecplot(folder + '/flowdata.tec', dat.grid)

# Save the results for the last time step using pyvista
lst.set_time(lst.times[-1])

df = pd.DataFrame(data=lst.element._data, columns=lst.element._col.keys())
cpg.cell_data['P(bar)'] = df['P(Pa)'] / barsa
cpg.cell_data['XCO2Liq'] = df['XCO2Liq']
# Exxagerate the z-coordinate for better visibility
#cpg = cpg.scale([1, 1, 5], inplace=True)
cpg.plot(scalars='P(bar)', show_edges=True)
cpg.save(folder + '/' + folder + '_z.vtk')

# Save the results for all time steps using pyTOUGH
# The results are visualized for the Cartesian geometry because the cell coordinates are not updated from CPG
pytough_vtk = False
if pytough_vtk:
    for t in lst.times:
        lst.set_time(t)
        lst.write_vtk(geo, folder + '.pvd', dat.grid)

    vtk_files = glob(folder + '*.pvd') + glob(folder + '*.vtu')
    for file in vtk_files:
        shutil.move(file, folder + '/' + file)

end_time = time.time()
dt_vtk = end_time - start_time
print("Saving VTK: %s sec" % dt_vtk)

# Saving the run times to csv
run_items = ['Setting up pyTOUGH parameters', 'Reading corner-point mesh', 'Creating MULgrid from CPG',
             'Setting up the grid in pyTOUGH', 'Running EXE', 'Fixing run results', 'Saving VTK results']
run_times = [dt_setup, dt_cpg, dt_mulgrid, dt_cpgrid, dt_exe, dt_fix, dt_vtk]
df = pd.DataFrame([run_items, run_times], index=['Operation', 'Run time (s)']).T
fname = folder + '/' + folder + '.csv'
df.to_csv(fname, sep=',', index=False)

print('Run times saved to ' + fname)
