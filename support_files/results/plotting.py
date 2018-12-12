## using evo to plot pose estimate file
from evo.core import metrics
from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)

import pprint
import numpy as np

from evo.tools import plot
import matplotlib.pyplot as plt
import csv

# temporarily override some package settings
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False
from evo.tools import file_interface

ref_file = "/home/soumya/vnav_proj_catkin_ws/src/Anticipated-VINS-Mono/support_files/results/data.csv"
est_file = "/home/soumya/vnav_proj_catkin_ws/src/Anticipated-VINS-Mono/support_files/results/poseEstimate.txt"

from evo.core import trajectory, geometry
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
##############
with open('poseEstimate.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    raw_mat = [row for row in plots]
error_msg = ("TUM trajectory files must have 8 entries per row "
             "and no trailing delimiter at the end of the rows (space)")
if len(raw_mat) > 0 and len(raw_mat[0]) != 8:
    print(error_msg)
mat = np.array(raw_mat).astype(float)
stamps = mat[:, 0]  # n x 1
xyz = mat[:, 1:4]  # n x 3
quat = mat[:, 4:]  # n x 4
quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
traj_est = PoseTrajectory3D(xyz, quat, stamps)
########################
traj_ref = file_interface.read_euroc_csv_trajectory(ref_file)

from evo.core import sync

max_diff = 0.01

traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

from evo.core import trajectory

traj_est_aligned = trajectory.align_trajectory(traj_est, traj_ref, correct_scale=False, correct_only_scale=False)

fig = plt.figure()
traj_by_label = {
    #"estimate (not aligned)": traj_est,
    "estimate (aligned)": traj_est_aligned,
    "reference": traj_ref
}
plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
plt.show()

### Now APE's Statistics
pose_relation = metrics.PoseRelation.translation_part
use_aligned_trajectories = True

if use_aligned_trajectories:
    data = (traj_ref, traj_est_aligned)
else:
    data = (traj_ref, traj_est)

ape_metric = metrics.APE(pose_relation)
ape_metric.process_data(data)
ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
print("Ape stat")
print(ape_stat)
ape_stats = ape_metric.get_all_statistics()
pprint.pprint(ape_stats)
seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]
fig = plt.figure()
plot.error_array(fig, ape_metric.error, x_array=seconds_from_start, statistics=ape_stats,
                 name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)")
plt.show()

plot_mode = plot.PlotMode.xy
fig = plt.figure()
ax = plot.prepare_axis(fig, plot_mode)
plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
plot.traj_colormap(ax, traj_est_aligned if use_aligned_trajectories else traj_est, ape_metric.error,
                   plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
ax.legend()
plt.show()


### Now RPE's Statistics


pose_relation = metrics.PoseRelation.rotation_angle_deg

# normal mode
delta = 1
delta_unit = metrics.Unit.frames

# all pairs mode
all_pairs = False  # activate
data = (traj_ref, traj_est_aligned)


rpe_metric = metrics.RPE(pose_relation, delta, delta_unit, all_pairs)
rpe_metric.process_data(data)

rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
print("RPE stat")
print(rpe_stat)
rpe_stats = rpe_metric.get_all_statistics()
pprint.pprint(rpe_stats)
# important: restrict data to delta ids for plot
import copy
traj_ref_plot = copy.deepcopy(traj_ref)
traj_est_aligned_plot = copy.deepcopy(traj_est_aligned)
traj_ref_plot.reduce_to_ids(rpe_metric.delta_ids)
traj_est_aligned_plot.reduce_to_ids(rpe_metric.delta_ids)
seconds_from_start = [t - traj_est_aligned.timestamps[0] for t in traj_est_aligned.timestamps[1:]]

fig = plt.figure()
plot.error_array(fig, rpe_metric.error, x_array=seconds_from_start, statistics=rpe_stats,
                 name="RPE", title="RPE w.r.t. " + rpe_metric.pose_relation.value, xlabel="$t$ (s)")
plt.show()

plot_mode = plot.PlotMode.xy
fig = plt.figure()
ax = plot.prepare_axis(fig, plot_mode)
plot.traj(ax, plot_mode, traj_ref_plot, '--', "gray", "reference")
plot.traj_colormap(ax, traj_est_aligned_plot, rpe_metric.error, plot_mode, min_map=rpe_stats["min"], max_map=rpe_stats["max"])
ax.legend()
plt.show()
