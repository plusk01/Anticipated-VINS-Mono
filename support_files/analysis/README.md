Analysis of Anticipated VINS-Mono
=================================

To analyze the performance of VINS-Mono with and without the attention and anticipation component, we use Michael Grupp's [`evo`](https://github.com/MichaelGrupp/evo) Python package. The code can be found in the `analysis.ipynb` Jupyter notebook (Python 2) found in this directory. Currently, the process is written expecting EuRoC comparisons.

## Generating ROSBag to be Analyzed

You can simply use the `bagoutput` arg when running your `roslaunch` command:

```bash
# be sure to source!
$ roslaunch vins_estimator euroc.launch sequence_name:=MH_01_easy bagoutput:=anticipated-MH1
```

This will save a bag to `/tmp/anticipated-MH1.bag`. Move this bag to this `analysis` directory to use the included Jupyter notebook for analysis.


### Related Literature

- Z. Zhang, D. Scaramuzza, **A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry**, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, 2018. [pdf](https://www.ifi.uzh.ch/dam/jcr:89d3db14-37b1-431d-94c3-8be9f37466d3/IROS18_Zhang.pdf) [toolbox](https://github.com/uzh-rpg/rpg_trajectory_evaluation) [comparison with `evo`](https://github.com/MichaelGrupp/evo/issues/112)
