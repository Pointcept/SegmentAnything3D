import argparse
from pathlib import Path
import shutil
import numpy as np
import json
import open3d as o3d
import torch

def save_mat_to_file(matrix: np.ndarray, filename: str):
    with open(filename, 'w') as f:
        for line in matrix:
            np.savetxt(f, line[np.newaxis], fmt='%f')

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment from 3D scanner app')
    dps_help = 'Point cloud data from App Store 3D scanner App https://apps.apple.com/us/app/id1419913995, '
    dps_help += 'which contains frame_X.jpg(s), depth_X.png(s), frame_X.json(s) and points.ply'
    parser.add_argument('--data_path', type=str, help=dps_help)

    sv_help = "Save path will be created and contains these things: color, depth, pose, intrinsics directory and points.pth."
    parser.add_argument('--save_path', type=str, help=sv_help)

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    data_dir = Path(args.data_path)
    frame_jpg_paths = data_dir.glob("frame_[0-9]*.jpg")
    frame_json_paths = data_dir.glob("frame_[0-9]*.json")
    depth_paths = data_dir.glob("depth_[0-9]*.png")


    # Generate output directories
    output_dir = Path(args.save_path)
    output_dir.mkdir(exist_ok=True)

    color_path = (output_dir / "color")
    color_path.mkdir(exist_ok=True)

    depth_path = (output_dir / "depth")
    depth_path.mkdir(exist_ok=True)

    intrinsics_path = (output_dir / "intrinsics")
    intrinsics_path.mkdir(exist_ok=True)

    pose_path = (output_dir / "pose")
    pose_path.mkdir(exist_ok=True)

    # Copy image files
    for pp in frame_jpg_paths:
        shutil.copy(pp, color_path)
    for pp in depth_paths:
        dest = depth_path / ("frame_" + pp.name[6:])
        shutil.copy(pp, dest)
    

    # Create and copy intrinsic_depth.txt
    first_json_path = next(iter(frame_json_paths))
    intrinsics_txt_path = intrinsics_path / "intrinsic_depth.txt"
    with open(str(first_json_path)) as f:
        jj = json.load(f)
    intrinsics = jj['intrinsics']
    intrinsics = np.array(intrinsics)
    intrinsics = intrinsics.reshape([3, 3]).astype(np.float32)
    rr = np.eye(4, dtype=np.float32)
    rr[0:3, 0:3] = intrinsics
    save_mat_to_file(rr, str(intrinsics_txt_path))

    # Pose
    for pp in frame_json_paths:
        with open(str(pp)) as f:
            jj = json.load(f)
        pose = jj['cameraPoseARFrame']
        pose = np.array(pose, dtype=np.float32).reshape([4, 4])
        dest = pose_path / ("frame_" + pp.name[6:-5] + ".txt")
        save_mat_to_file(pose, str(dest))

    # .ply to .pth
    ply_path = data_dir / "points.ply"
    pcd = o3d.io.read_point_cloud(str(ply_path))
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    save_dict = dict(coord=coords, color=colors, scene_id=data_dir.stem)
    dest = output_dir / "points.pth"
    torch.save(save_dict, str(dest))
    pass

if __name__ == '__main__':
    main()
    pass