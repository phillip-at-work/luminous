# luminous

An object-oriented ray trace library with a scriptable API. See repo `https://github.com/phillip-at-work/luminous-examples` to explore the most up-to-date features.

# TODO roadmap (priority where numbered)
0) DONE. add optional raydebugger as null pattern to map rays with vtk
1) DONE. add raydebugger rays from detector to elements, elements to source
2) DONE. allow rays to both transmit and reflect
3) real spectral model, e.g., sources can model any broadband spectrum (currently rgb)
4) DONE. support for multiple sources
5) DONE. support for multiple detectors
6) DONE. abstract color model from Scene to Detector ABC
7) multi-thread or multi-process support for multiple detectors
8) add support for specific source ray geometries (standard parameterized implementations for laser, isotropic source, etc.)
9) reformulate ray trajectories from detector such as to assume a pinhole or lens assembly (currently transmits along the normal)
- add optional ray time of flight model
- atmospheric models/ media in aggregate
- add triangular mesh elements (arbitrary closed polygons, possibly as STLs)
- DONE. Camera object basic refraction and reflection model
- add Lens element defined by two radii and center thickness
- add raytrace animation option for detectors, elements, and sources. to support time-functional emissions (radar).
- add concrete implementation for PowerMeter (Detector ABC)
- polarization model for sources
- permit Detector to accept a lens assembly argument (Detector should be capable of tracing rays w/o a lens)
- add STL filetype saving for ray debugger
- allow ray debugger to plot elements as well as rays (currently)
- basic source and detector model for dipole antenna (radar)
- hard code frasnel equations into Scene. assumes unpolarized sources when polarization not specified.