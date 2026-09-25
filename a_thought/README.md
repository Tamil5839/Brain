# A THOUGHT

A film of a whole-brain simulation of the fruit fly, built on the complete published
wiring diagram (FlyWire release 783). Every neuron sits at its true position and every
flash of light is a spike from the simulation, a leaky integrate-and-fire model that
reproduces Shiu et al. (2024, *Nature*): sugar taste drives the feeding motor neuron MN9,
bitter does not, and bitter suppresses the sugar response.

*A simulation built on the real wiring of a fruit fly's brain.*

See **[NOTES.md](NOTES.md)** for sources, versions, checksums, model parameters,
validation results and every visual-aid disclosure.

## Pipeline

| step | command | output |
|---|---|---|
| download + manifest | `python fetch.py` | `data/raw/`, `data/manifest.json` |
| clean + assemble | `python prepare.py` | `data/processed/`, `results/prepare_log.json` |
| experiments A–D | `python -m sim.run` | `results/spikes/*.npz`, `results/experiments.json` |
| pathway + skeletons | `python -m render.pathway` | `results/pathway.json` |
| validation (section 5) | `python -m sim.validate` | `results/validation.json` |
| film, 4K60 | `python -m render.film master` | `out/a_thought.mp4` |
| film, 1080p | `python -m render.film 1080p` | `out/a_thought_1080p.mp4` |
| vertical cut | `python -m render.film vertical` | `out/a_thought_vertical.mp4` |
| poster | `python -m render.film poster` | `out/poster.png` |
| notes | `python notes.py` | `NOTES.md` |
| tests | `python -m pytest tests/` | |

Rendering refuses to start unless `results/validation.json` reports that every test passed.
The Brian2 reference runs (`sim/brian2_reference.py`, `tests/brian2_*equivalence.py`) need
the authors' environment: Brian2 2.5.1 with NumPy 1.24 in a separate virtual environment
(`requirements-brian2.txt` has the pinned versions and install steps).

Requirements: Python 3.11, `requirements.txt`, ffmpeg, Mesa EGL (`libegl1`, `libegl-mesa0`)
and the Inter font (`fonts-inter`). A GPU is used if present; without one, Mesa's llvmpipe
renders on the CPU (about 0.8 s per 4K frame on 4 cores). `A_THOUGHT_RENDERER=cpu` selects a
slow NumPy fallback for previews on machines without any OpenGL.

## Layout

```
fetch.py, prepare.py            data download, manifest, cleaning
sim/model.py                    the LIF model (Brian2 semantics of the authors' code)
sim/run.py, sim/validate.py     experiments, section-5 tests and cross-checks
sim/networks.py                 network variants (authors' v783, v630, flypoke settings)
sim/brian2_reference.py         runs the authors' unmodified Brian2 code
render/gl.py, render/cpu.py     OpenGL renderer and CPU fallback
render/scene.py                 camera, colours, neuron attributes
render/film.py                  timeline, captions, encoding
render/pathway.py, fly.py,      pathway selection + skeleton reader, fly illustration,
render/overlay.py, poster.py    text overlays, poster
tests/                          pytest suite + Brian2 equivalence scripts
```
