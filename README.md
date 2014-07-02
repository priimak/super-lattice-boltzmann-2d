Superlattice 2D Boltzmann Solver
================================

Overview
--------
Code in this repository implements finite differences method for solution of Boltzmann 
equation for electron transport in two dimensional semiconductor super lattices. 
For reference see this article http://arxiv.org/abs/1401.6047 that contains mathematical 
model for this system and description of the paramaters.
It contains several implementations. Two implementations in targeted 
to common CPU. One single threaded (bin/boltzmann_c_solver) and another one uses naive 
OpenMP (bin/boltzmann_openmp_solver). Other
implementations are in CUDA and where tested on GTX680 (bin/boltzmann_solver). 
They are distinguished 
by different versions of kernel. To activate one or another kernel they have to
be re-complied with environment variable BLTZM_KERNEL indicated kernel version.
BLTZM_KERNEL may take one of the following values

* 1	     - One thread per lattice point.
* 2          - One thread per lattice point. Uses shared memory.

In all following kernels each thread computes lattice values for each <em>n</em>.
* 310	     - One thread per <em>n</em>.
* 311 	     - -//- and removed divergent flows.
* 321 	     - -//- and loops unrolled twice.
* 341	     - -//- and loops unrolled four times.
* 342	     - --------------//---------------- and elements partially reused.
* 4	     - Loops are staggered and elemenets reused. Not unrolled.

Usage
-----
Following parameters can be specified when executing boltzmann_solver.
All parameters valuess are passed through use '=' sign. For example display=77

* **display**    - indicates type of the output generated. Possible values are

    - 4           - compute instantaneous and averaged over external a/c emf. period drift velocities and absorption. Output file will contain detailed description of each field.

    - 7           - generate sequence of frame data files (files frame%08d.data) starting from time 'frame-start', which can be used to create movie sequences. 

    - 8           - at the end of simulation generate single shot frame data file frame.data

    - 77          - similar to case of display=7 but instead of frame data produce output similar to display=4 containing time evolution of instantaneous drift velocities etc.

* **n-harmonics** - Maximum number of harmonics in expansion along the $\phi_x$ axis.

* **PhiYmin**     - $\phi_y$ region is infinite, but to be able to numerically solve this problem we have to limit it to values from PhiYmin to PhiYmax

* **PhiYmax**     - $\phi_y$ region is infinite, but to be able to numerically solve this problem we have to limit it to values from PhiYmin to PhiYmax

* **t-max**       - computation runs from time t=0 to (t-max)+2*PI/omega

* **frame-start** - if you are making movie (display=7) frame data (files frame%08d.data) files start generating from time 'frame-start'

* **dt**          - time step 

* **g-grid**      - number of cells alone the $\phi_y$ axis, with step size (PhiYmax - PhiYmin)/g-grid

* **quiet**       - suppress progress output

* **device**      - if you have more that one video card you may choose a specific one to run on by using this parameter, they are numbered starting from 0 (default value)

* **o**           - output file. You can also use '+' sign to append to the existing file, like so o=+output.data

Following parameters have exact corresponance in http://arxiv.org/abs/1401.6047 
Here we are showing to what LaTeX source each one corresponds to

* **E_dc**       - corresponds to $E_{dc}$

* **E_omega**    - corresponds to $E_{\omega}$

* **omega**      - corresponds to $\omega$

* **mu**         - corresponds to $\m$

* **alpha**      - corresponds to $\alpha$

* **B**          - corresponds to $B$

