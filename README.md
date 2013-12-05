super-lattice-boltzmann-2d
==========================

Overview
--------
Code in this repository implements finite differences method for solution of Boltzmann equation for electron transport in two dimensional semiconductor super lattices. It contains several implementations. Two implementations in targeted 
to common CPU. One single threaded and another one uses naive OpenMP. Other
implementations are in CUDA and where tested on GTX680. They are distinguished 
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

