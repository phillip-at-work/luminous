# luminous

An object-oriented ray trace library with a scriptable API. See repo `https://github.com/phillip-at-work/luminous-examples` to explore the most up-to-date features.

## Running Tests

```bash
# Normal mode: simple list of tests with pass/fail
./run_tests.sh

# Debug mode: shows print statements even when tests pass
./run_tests.sh --debug

# Coverage report
./run_tests.sh --coverage
```

## Reverse Raytrace

- Intended largely as a debug feature to confirm that scene elements are place appropriately
- Does not factor in Fresnel losses. Instead, the reverse trace uses an arbitrary counter to limit the number of rays the reflect and transmit. This can sometimes produce misleading results, as those rays will necessarily progagate some constant count, rather than diminishing in intensity from the source.
- Cannot image sources. e.g., sources in the detector's view are invisible
- Does support simultaneous reflection and transmission for transparent volumes

## Trace Log

`Scene` records the traversal it performs - one row per decision - as an aligned
table. The log file IS that table: the same bytes in the file, on the terminal, and
in the rows the tests assert against. There is no second format to turn on.

Logging is off by default (`log_level=logging.CRITICAL`), and the tracer checks the
level before it builds a row, so a scene that does not log pays nothing for the
feature. The log file is truncated when the `Scene` is created, so the file is
exactly the trace of that scene, not an append log of every run.

```python
import logging
from src.math.vector import Vector
from src.scene.scene import Scene
from src.element.element import SphereElement
from src.element.source import IsotropicSource
from src.element.detector import Camera
from src.utilities.logconfig import TERMINATION, SHADERS, read_log_table

scene = Scene(log_level=logging.DEBUG, log_file="./results/luminous.log")
scene += IsotropicSource(node_id="SUN", center=Vector(0, 6, 1), radius=0.05, color=Vector(1, 0, 0))
scene += Camera(node_id="CAMERA", width=500, height=500, position=Vector(0, 1, 0),
                pointing_direction=Vector(0, 0, 1), screen_width=1e-6, screen_height=1e-6)
scene += SphereElement(node_id="GLASS_1", center=Vector(0, 0.999999, 3), radius=0.5e-6,
                       color=Vector(0, 0.2, 0), refractive_index=1.5, transparent=True,
                       user_params={'specular': 0.25, 'n_s': 10})
scene.raytrace()

scene.show_trace()                        # the whole table, printed
scene.show_trace(types=[TERMINATION])     # filtered by TYPE token

text = read_log_table(scene.log_file,     # the same text, as a string
                      types=[SHADERS])
```

The same reader is a CLI over any log file, no API involved:

```bash
python -m src.utilities.logconfig ../results/luminous.log --types TERMINATION
```

The opening rows of a real 250,000-ray trace - 238 rows in the file - and its first
termination:

```text
TYPE         from              target            action              hop  b   d  n       src  path
ORIGIN       CAMERA            CAMERA            init_trace                      250000
LEG          CAMERA            START             reverse_leg         1    0   0               CAMERA
CONTEXT      CAMERA            SUN               query_visibility    1    0   0  250000
TERMINATION  CAMERA            INFINITY          ray_escaped         1    0   0  236126
```

`from:` answers "where did this leg most recently originate?" and `target:` answers
"which surface or object did this iteration occur on?". With `TYPE` and `action`
those four identify a single decision in the path, which is what makes the file
diffable line-for-line.

Both name a node by its `node_id`, which is a **required** constructor argument for
every `Element`, `Source` and `Detector`:
`Camera(node_id="CAMERA", ...)`, `SphereElement(node_id="GLASS_1", ...)`. Nothing
mints a name on your behalf, so the tokens in a log are the tokens of the program that
wrote it. An id is validated by `validate_node_id` (`src/utilities/logconfig.py`,
beside the format it protects): letter-initial, then letters, digits, `_`, `-`,
because the characters that carry structure in `path` - `>`, `(`, `)`, `=` - cannot
appear in an id. It must be unique within a `Scene` - `path` is parsed to recover the
node a leg departed from, so two nodes sharing an id would resolve a hop to the wrong
object.

- `n:` is how many rays a row describes. A record covers a whole ray mask, not one
  ray, which is why tracing 250,000 pixels is ~240 rows and not 250,000.
- `src:` is how many sources are visible from that vertex, so `src: 0` is shadow -
  the answer to "is this surface lit from here?" without reading the path.
- A blank cell is a field that record omitted; `0` is a value and is printed. A blank
  `src:` means the row says nothing about visibility, `src: 0` means it says shadow.
- `hop:`, `b:`, `d:` are the budgets being spent: the global recursion counter,
  bounce depth, and transmission depth inside transparent volumes.
- `path` is a breadcrumb of the traversals that produced the row, and is never
  truncated - it is the traversal record.

Where the details live, and why they are split:

- **What each row means** - the row-by-row legend of the `TYPE` vocabulary,
  including why `emission_model` fires only on the `START` leg - is the module
  docstring of `tests/test_reverse_trace.py`, beside the known sequences it
  explains.
- **How the format is built** - the rules that keep a row's bytes dependent only on
  that row (widths declared in `LOG_COLUMNS`, never measured from the data; `path`
  last and unbounded; nothing truncated; every line `rstrip`-ed; a blank cell is an
  omitted field, never a zero) - is the module docstring of
  `src/utilities/logconfig.py`.
- **An end-to-end example** is `refraction_example_mini.py` in the examples repo: it
  traces at `DEBUG` and prints the terminations, and writes `results/luminous.log`.

Column widths are deliberately not repeated here. They move whenever a token is
renamed, and a duplicated table goes stale on the first rename - the declared
widths in `LOG_COLUMNS` are the only authority. The tests assert these rows
verbatim, so changing a width means re-laying the expected rows in
`tests/test_reverse_trace.py`.

`VERBOSE_LOG_FORMAT` (pass `log_format=` to `setup_logging`) prefixes wall-clock
time onto each row. Its prefix shifts every column off its declared offset, so it is
for reading with the eye only, never for a log a test asserts against.

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