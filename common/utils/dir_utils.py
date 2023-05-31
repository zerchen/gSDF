import os
import sys
import json

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def export_pose_results(path, pose_result, metas):
    if isinstance(pose_result, list):
        num_frames = len(pose_result)
        for i in range(num_frames):
            sample_id = metas['id'][i][0]
            with open(os.path.join(path, sample_id + '.json'), 'w') as f:
                output = dict()
                output['cam_extr'] = metas['cam_extr'][i][0][:3, :3].cpu().numpy().tolist()
                output['cam_intr'] = metas['cam_intr'][i][0][:3, :3].cpu().numpy().tolist()
                if pose_result[i] is not None:
                    for key in pose_result[i].keys():
                        if pose_result[i][key] is not None:
                            output[key] = pose_result[i][key][0].cpu().numpy().tolist()
                        else:
                            continue
                json.dump(output, f)
    else:
        sample_id = metas['id'][0]
        with open(os.path.join(path, sample_id + '.json'), 'w') as f:
            output = dict()
            output['cam_extr'] = metas['cam_extr'][0][:3, :3].cpu().numpy().tolist()
            output['cam_intr'] = metas['cam_intr'][0][:3, :3].cpu().numpy().tolist()
            if pose_result is not None:
                for key in pose_result.keys():
                    if pose_result[key] is not None:
                        output[key] = pose_result[key][0].cpu().numpy().tolist()
                    else:
                        continue
            json.dump(output, f)