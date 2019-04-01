from mvsnet import preprocess as pp
import json
import os


def depth_pfm_to_png(pfm_path, png_path):
    """Reads in a depth map in .pfm format from pfm_path
    and writes a depth map in .png format at png_path"""
    depth = pp.load_pfm(open(pfm_path))
    pp.write_depth_map(png_path, depth)


def cam_to_json(txt_path, json_path, scale_factor = 1.0):
    """ Converts the MVSNet cam txt file into the camera.json
    format prepared by export_densify_frames. scale_factor can be used to
    rescale the cams from the originals
    """
    #print('Loading camera from',txt_path)
    #print('Writing camera to', json_path)
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

def pair_to_covisibility(pair_path, output_path, min_depth = 400.0, max_depth=1000.0 ):
    """ Converts the MVSNet pair.txt file, into the covisibility.json format """
    lines = [line.strip() for line in open(pair_path)]
    covis = {}
    for i in range(2,len(lines), 2):
        cluster = {}
        data = lines[i].split()
        key = str(data[1])
        views = []
        for j in range(3,len(data),2):
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
    image_index +=1
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


    


