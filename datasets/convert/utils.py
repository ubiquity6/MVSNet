# from mvsnet import preprocess as pp
import numpy as np
import imageio
import json
import glob
import os


def depth_pfm_to_png(pfm_path, png_path):
    """Reads in a depth map in .pfm format from pfm_path
    and writes a depth map in .png format at png_path"""
    depth = pp.load_pfm(open(pfm_path))
    pp.write_depth_map(png_path, depth)


def cam_to_json(txt_path, json_path, scale_factor=1.0):
    """ Converts the MVSNet cam txt file into the camera.json
    format prepared by export_densify_frames. scale_factor can be used to
    rescale the cams from the originals
    """
    # print('Loading camera from',txt_path)
    # print('Writing camera to', json_path)
    cam = pp.load_cam_from_path(txt_path, interval_scale=1.0, max_d=0.0)
    cam_json = {}
    # Add intrinsics
    intrin = {}
    intrin["fx"] = cam[1, 0, 0] * scale_factor
    intrin["fy"] = cam[1, 1, 1] * scale_factor
    intrin["px"] = cam[1, 0, 2] * scale_factor
    intrin["py"] = cam[1, 1, 2] * scale_factor
    cam_json["intrinsics"] = intrin
    # Add extrinsics
    mat = {}
    for row in range(4):
        for col in range(4):
            key = "{},{}".format(row, col)
            elem = cam[0, row, col]
            # Convert the translation elements from mm to m
            if col == 3 and row != 3:
                elem = elem / 1000
            mat[key] = elem
    pose = {}
    pose["matrix"] = mat
    cam_json["pose"] = pose
    with open(json_path, 'w') as f:
        json.dump(cam_json, f)


def pair_to_covisibility(pair_path, output_path, min_depth=400.0, max_depth=1000.0):
    """ Converts the MVSNet pair.txt file, into the covisibility.json format """
    lines = [line.strip() for line in open(pair_path)]
    covis = {}
    for i in range(2, len(lines), 2):
        cluster = {}
        data = lines[i].split()
        key = str(data[1])
        views = []
        for j in range(3, len(data), 2):
            views.append(int(data[j]))
        cluster["views"] = views
        cluster["min_depth"] = min_depth
        cluster["max_depth"] = max_depth
        covis[key] = cluster
    with open(output_path, 'w') as f:
        json.dump(covis, f)


def image_name(image_index, lighting_index):
    """ Returns the name of the dtu image. image_index goes from 0-48
    and lighting_index goes from 0 to 6 """
    image_index += 1
    if image_index < 10:
        return 'rect_00{}_{}_r5000.png'.format(image_index, lighting_index)
    else:
        return 'rect_0{}_{}_r5000.png'.format(image_index, lighting_index)


def depth_name(depth_index):
    """ Returns the name of the depth map for index: 0-48 """
    if depth_index < 10:
        return 'depth_map_000{}.pfm'.format(depth_index)
    else:
        return 'depth_map_00{}.pfm'.format(depth_index)


def cam_name(cam_index):
    """ Returns the name of the depth map for index: 0-48 """
    if cam_index < 10:
        return '0000000{}_cam.txt'.format(cam_index)
    else:
        return '000000{}_cam.txt'.format(cam_index)


def list_no_hidden(d):
    """ Lists all elements of a directory except for hidden elements
    that being with a period. """
    return [f for f in os.listdir(d) if not f.startswith('.')]


def cameras_from_demon(d, scale_factor=1.0):
    """  Converts camera data from the DPSNet training format to
    DeMoN data to mvs-training format
    Args:
        d: the directory containing DeMoN sessions
        formatted by DPSNet formatter
    """
    intrinsics = np.genfromtxt(os.path.join(d, 'cam.txt'))
    poses = np.genfromtxt(os.path.join(d, 'poses.txt'))
    num_cams = poses.shape[0]
    camera_dir = os.path.join(d, 'cameras')
    os.makedirs(camera_dir, exist_ok=True)
    for i in range(num_cams):
        json_path = os.path.join(camera_dir, '{}.json'.format(i))
        cam_json = {}
        # Add intrinsics
        intrin = {}
        intrin["fx"] = intrinsics[0, 0] * scale_factor
        intrin["fy"] = intrinsics[1, 1] * scale_factor
        intrin["px"] = intrinsics[0, 2] * scale_factor
        intrin["py"] = intrinsics[1, 2] * scale_factor
        cam_json["intrinsics"] = intrin
        # Add extrinsics
        mat = {}
        for row in range(3):
            for col in range(4):
                key = "{},{}".format(row, col)
                mat[key] = poses[i, row*4 + col]
        mat["3,0"] = 0.0
        mat["3,1"] = 0.0
        mat["3,2"] = 0.0
        mat["3,3"] = 1.0
        pose = {}
        pose["matrix"] = mat
        cam_json["pose"] = pose
        with open(json_path, 'w') as f:
            json.dump(cam_json, f)
    return num_cams


def depths_from_demon(d):
    """  Converts depth data from the DPSNet training format to
DeMoN data to mvs-training format
Args:
    d: the directory containing DeMoN sessions
    formatted by DPSNet formatter
"""
    depth_paths = sorted(glob.glob(os.path.join(d, '*.npy')))
    depths_dir = os.path.join(d, 'depths')
    os.makedirs(depths_dir, exist_ok=True)
    max_depth = 0.0
    min_depth = 100000.0
    for i, p in enumerate(depth_paths):
        dpath = os.path.join(depths_dir, '{}.png'.format(i))
        data = np.load(p)
        # convert to mm
        data *= 1000.0
        data = np.clip(data, 0, 65535).astype(np.uint16)
        imageio.imsave(dpath, data)
        dmax = data[data != 65535].max()
        # ignore zero values when filtering
        dmin = data[data != 0].min()
        if dmax > max_depth:
            max_depth = dmax
        if dmin < min_depth:
            min_depth = dmin
        os.remove(p)
    return len(depth_paths), min_depth, max_depth


def images_from_demon(d):
    """  Converts images from the DPSNet training format to
DeMoN data to mvs-training format
Args:
    d: the directory containing DeMoN sessions
    formatted by DPSNet formatter
"""
    image_paths = sorted(glob.glob(os.path.join(d, '*.jpg')))
    images_dir = os.path.join(d, 'images')
    os.makedirs(images_dir, exist_ok=True)
    for i, p in enumerate(image_paths):
        ipath = os.path.join(images_dir, '{}.jpg'.format(i))
        os.rename(p, ipath)

    return len(image_paths)


def covisibility_from_demon(d, min_depth=400.0, max_depth=65535.0):
    covis_path = os.path.join(d, 'covisibility.json')
    covis = {}
    num = len(glob.glob(os.path.join(d, 'depths', '*.png')))
    # We add all cyclic permutations of elements, with each one
    # being the reference image for a given cluster
    v = range(num)
    for i in range(num):
        views = [x for x in v if x != i]
        cluster = {}
        cluster["views"] = views
        cluster["min_depth"] = float(min_depth)
        cluster["max_depth"] = float(max_depth)
        covis[str(i)] = cluster
    with open(covis_path, 'w') as f:
        json.dump(covis, f)
