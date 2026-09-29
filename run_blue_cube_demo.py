# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
from estimater import *
from datareader import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/cubes/mesh/blue_cube.obj')
    parser.add_argument('--scene_dir', type=str, default=f'{code_dir}/demo_data/cubes')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    mesh = trimesh.load(args.mesh_file)

    # If the mesh has no UV coords (e.g. a plain .obj with no texture),
    # FoundationPose will crash. Apply a simple spherical unwrap as a fallback.
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        logging.info("Mesh has no UVs — applying spherical unwrap")
        mesh = mesh.unwrap()
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("Estimator initialised")

    reader = YcbineoatReader(video_dir=args.scene_dir, shorter_side=None, zfar=np.inf)

    # Set up mp4 writer — we'll grab dimensions from the first frame
    first_frame = reader.get_color(0)
    h, w = first_frame.shape[:2]
    video_path = os.path.join(debug_dir, 'tracking.mp4')
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    logging.info(f"Video will be saved to {video_path}")

    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            mask_path = os.path.join(args.scene_dir, 'masks', '0000.png')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f'No mask found at {mask_path}. '
                    f'Run generate_mask.py first.'
                )
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            logging.info("Initial pose registered")

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('blue cube tracking', vis[..., ::-1])
            cv2.waitKey(1)
            writer.write(vis[..., ::-1])  # vis is RGB, writer expects BGR

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    writer.release()
    logging.info(f"Video saved to {video_path}")
