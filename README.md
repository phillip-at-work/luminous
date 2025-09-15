# luminous

An object-oriented ray trace library with a scriptable API. See repo `https://github.com/phillip-at-work/luminous-examples` to explore the most up-to-date features.

# TODO roadmap (priority where numbered)
0) DONE. add optional raydebugger as null pattern to map rays with vtk
1) DONE. add raydebugger rays from detector to elements, elements to source
2) refraction model, e.g., rays can both transmit and reflect
3) real spectral model, e.g., sources can model any broadband spectrum (currently rgb)
4) DONE. support for multiple sources (currently only a single point source)
5) DONE. support for multiple detectors
6) DONE. abstract color model from Scene to Detector ABC
7) multi-thread or multi-process support for multiple detectors
8) add support for specific source ray geometries (standard parameterized implementations for laser, isotropic source, etc.)
- add optional ray time of flight model
- atmospheric models/ media in aggregate
- add triangular mesh elements (arbitrary closed polygons)
- add Lens element defined by two radii and center thickness
- add raytrace animation option for detectors, elements, and sources (independent)
- add concrete implementation for PowerMeter (Detector ABC)
- polarization model for sources (also allows refraction/reflection model to use polarization)
- permit Detector to accept a lens assembly argument (Detector should be capable of tracing rays w/o a lens)
