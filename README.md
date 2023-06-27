# MITWake

MITWake is a Python package that implements a static wake model used at Howland Lab. It provides features for modeling wake deflection and the induction-yaw behavior of an actuator disk as described in [Heck et al. (2023)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/modelling-the-induction-thrust-and-power-of-a-yawmisaligned-actuator-disk/3A34FC48A6BC52A78B6D221C13F4FC3A?utm_campaign=shareaholic&utm_medium=copy_link&utm_source=bookmark)[^fn1]. The package also includes several wake superposition models ([Howland et al. (2016)](https://www.pnas.org/doi/full/10.1073/pnas.1903680116)[^fn2]) and supports analytical gradients with respect to yaw angle and thrust for each turbine.

## Getting Started

MITWake is compatible with Python versions 3.7.1 to 3.9.

### Installation

To install MITWake, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/Howland-Lab/MITWake.git
```

2. Navigate to the MITWake directory and install the package using `pip`:
```bash
cd MITWake
pip install .
```

### Usage

For usage examples, refer to the example scripts provided in the `examples` folder.

Documentation can be found at [howland-lab.github.io/MITWake/](https://howland-lab.github.io/MITWake/).

# References
[^fn1]: Heck, Kirby S., Hannah M. Johlas, and Michael F. Howland. "[Modelling the induction, thrust and power of a yaw-misaligned actuator disk](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/modelling-the-induction-thrust-and-power-of-a-yawmisaligned-actuator-disk/3A34FC48A6BC52A78B6D221C13F4FC3A?utm_campaign=shareaholic&utm_medium=copy_link&utm_source=bookmark)." Journal of Fluid Mechanics 959 (2023): A9.
[^fn2]: Howland, Michael F., Sanjiva K. Lele, and John O. Dabiri. "[Wind farm power optimization through wake steering](https://www.pnas.org/doi/full/10.1073/pnas.1903680116)." Proceedings of the National Academy of Sciences 116.29 (2019): 14495-14500.
