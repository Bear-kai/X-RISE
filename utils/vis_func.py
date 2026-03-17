import numpy as np


def visualize(vis, *pcs, **kwargs):
    vis_pc = np.concatenate(pcs)
    vis_label = np.ones((sum([p.shape[0] for p in pcs])), np.int64)
    a = 0
    for i, pc in enumerate(pcs):
        vis_label[a:a+pc.shape[0]] = i + 1
        a += pc.shape[0]
    # vis.scatter(vis_pc, vis_label, **kwargs)
    num_axis_pts = 50
    x_axis = np.stack([np.array([1,0,0])*v for v in np.linspace(0.01, 1, num_axis_pts)])
    y_axis = np.stack([np.array([0,1,0])*v for v in np.linspace(0.01, 1, num_axis_pts)])
    z_axis = np.stack([np.array([0,0,1])*v for v in np.linspace(0.01, 1, num_axis_pts)])
    xyz_axis = np.concatenate([x_axis, y_axis, z_axis])
    x_color = np.stack([np.array([255,0,0])] * num_axis_pts)
    y_color = np.stack([np.array([0,255,0])] * num_axis_pts)
    z_color = np.stack([np.array([0,0,255])] * num_axis_pts)
    colors = np.concatenate([x_color, y_color, z_color])		# int64
    if "markercolor" in kwargs['opts'].keys():
        kwargs['opts']["markercolor"] = np.concatenate([(kwargs['opts']["markercolor"]).astype(np.int64), colors])
    vis_pc = np.concatenate([vis_pc, xyz_axis])
    vis_label = np.concatenate([vis_label, np.ones((num_axis_pts*3), np.int64)*(i+2)])
    vis.scatter(vis_pc, vis_label, **kwargs)


def gen_obj_frame_colored_pcd(trans, rot, scale=0.5):
    num_axis_pts = int(round(50*scale))
    x_axis = np.stack([np.array([scale,0,0])*v for v in np.linspace(0.01, 1, num_axis_pts)])
    y_axis = np.stack([np.array([0,scale,0])*v for v in np.linspace(0.01, 1, num_axis_pts)])
    z_axis = np.stack([np.array([0,0,scale])*v for v in np.linspace(0.01, 1, num_axis_pts)]) 
    xyz_axis = np.concatenate([x_axis, y_axis, z_axis])         # (3n,3)
    x_color = np.stack([np.array([255,0,0])] * num_axis_pts)
    y_color = np.stack([np.array([0,255,0])] * num_axis_pts)
    z_color = np.stack([np.array([0,0,255])] * num_axis_pts)
    colors = np.concatenate([x_color, y_color, z_color])
    xyz_axis = xyz_axis @ rot.T + trans
    return xyz_axis, colors


def random_down_sample_pcd(pts, colors=None, sampling_ratio=0.2):
    total_num = len(pts)
    sample_num = int(total_num * sampling_ratio)
    sample_inds = np.random.choice(np.arange(total_num), sample_num, replace=False)
    if colors is not None:
        return pts[sample_inds], colors[sample_inds]
    return pts[sample_inds]