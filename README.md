# Echellogram Plot

A small script to generate nice-looking plots showing the spectral layout of an echelle spectrograph. Key stellar spectral lines are shown, and the spectral orders are labelled with the corresponding m-number and central wavelength.

Dependencies:
- `numpy`
- `matplotlib`
- `PyZDDE` (https://github.com/xzos/PyZDDE)
- `PyEchelle` (https://gitlab.com/Stuermer/pyechelle)
- For custom plot style only:
  - `cycler`
  - `palettable`

Example plot generated using this script:

![Example plot](https://github.com/pemberj/echellogram_plot/blob/main/example_echellogram_plot.png?raw=true)
