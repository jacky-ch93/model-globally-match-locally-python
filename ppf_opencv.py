import argparse
import cv2 as cv
import open3d as o3d
import numpy as np
import copy


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="Path to the model pointcloud")
    parser.add_argument("--scene", help="Path to the scene pointcloud")
    parser.add_argument("--topN", help="the number of results to be selected")
    parser.add_argument(
        "--fast", action="store_true", help="Use the c++ extension for speeeeeed"
    )
    parser.add_argument(
        "--scene-pts-fraction",
        default=0.2,
        type=float,
        help="Fraction of scene points to use as reference",
    )
    parser.add_argument(
        "--ppf-num-angles",
        default=30,
        type=int,
        help="Number of angle steps used to discretize feature angles.",
    )
    parser.add_argument(
        "--ppf-rel-dist-step",
        default=0.05,
        type=float,
        help="Discretization step of feature distances, relative to model diameter.",
    )
    parser.add_argument(
        "--alpha-num-angles",
        default=30,
        type=int,
        help="Number of angle steps used to discretize the rotation angle alpha.",
    )
    parser.add_argument(
        "--cluster-max-angle",
        type=float,
        default=30,
        help="Maximal angle between poses after which they don't belong to same cluster anymore. [degrees]",
    )
    args = parser.parse_args()
    
    try:
        src_model = o3d.io.read_triangle_mesh(args.model)
        # src_model.remove_degenerate_triangles()
        # src_model.remove_duplicated_triangles()
        # src_model.remove_duplicated_vertices()
        # src_model.remove_non_manifold_edges()
        # src_model.compute_triangle_normals(normalized=True)
        model_icp = src_model.sample_points_uniformly(number_of_points=50000)
    except RuntimeError:
        src_model = o3d.io.read_point_cloud(args.model)
        model_icp = src_model.farthest_point_down_sample(num_samples=50000)

    try:
        src_scene = o3d.io.read_triangle_mesh(args.scene)
        # src_scene.remove_degenerate_triangles()
        # src_scene.remove_duplicated_triangles()
        # src_scene.remove_duplicated_vertices()
        # src_scene.remove_non_manifold_edges()
        # src_scene.compute_triangle_normals(normalized=True)
        scene_icp = src_scene.sample_points_uniformly(number_of_points=50000)
    except RuntimeError:
        src_scene = o3d.io.read_point_cloud(args.scene)
        scene_icp= src_scene.farthest_point_down_sample(num_samples=50000)
        # down_indices = src_scene.get_indices_from_mask(indices)

    # process normals of pointclouds:
    # if model.normals:
    model_icp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=20))
    model_icp.normalize_normals()
    model = model_icp.farthest_point_down_sample(num_samples=200)
    model_points = np.asarray(model_icp.points).astype('float32')
    model_normals = np.asarray(model_icp.normals).astype('float32')
    model_opencv = np.column_stack([model_points, model_normals])
    # if scene.normals:
    scene_icp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=20))
    """
    for index in range(len(scene_icp.normals)):
        scene_icp.normals[index] = -1 * scene_icp.normals[index]
    """
    if "xinhua" in args.model:
        for index in range(len(scene_icp.points)):
            if (scene_icp.points[index].dot(scene_icp.normals[index])) > 0:
                scene_icp.normals[index] = -1 * scene_icp.normals[index]

    scene_icp.normalize_normals()
    scene = scene_icp.farthest_point_down_sample(num_samples=100)
    scene_points = np.asarray(scene_icp.points).astype('float32')
    scene_normals = np.asarray(scene_icp.normals).astype('float32')
    scene_opencv = np.column_stack([scene_points, scene_normals])

    # visualize source(model) and traget(scene) pointcloud
    _model_vis = copy.deepcopy(src_model)
    _scene_vis = copy.deepcopy(src_scene)
    _model_vis.paint_uniform_color([1, 0.5, 0])
    _scene_vis.paint_uniform_color([0.5, 1, 0])
    # o3d.visualization.draw_geometries([model_icp, scene_icp], point_show_normal=True, width=1280, height=760, window_name="source(model) and traget(scene) pointcloud")
    
    # model = cv.ppf_match_3d.loadPLYSimple("./example_models/nantong-xinhua-1.ply", 0)
    ppf = cv.ppf_match_3d.PPF3DDetector(relativeSamplingStep=0.05, relativeDistanceStep=0.05, numAngles=30)
    
    ppf.trainModel(model_opencv)
    # ppf.write("./1.yaml")
    # scene = cv.ppf_match_3d.loadPLYSimple("./example_models/nantong-xinhua-1-move.ply", 0)
    results = ppf.match(scene_opencv, 0.05, 0.05)
    ICP = cv.ppf_match_3d.ICP(100, 0.05, 2, 8)
    retval, poses = ICP.registerModelToScene(model_opencv, scene_opencv, results)
    for pose_i in poses:
        print(pose_i.pose)
    for result in results:
        print(result.pose)
    a = 0

if __name__ == "__main__":
    main()
