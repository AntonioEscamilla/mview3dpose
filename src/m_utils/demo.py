import os
import os.path as osp
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if __name__ == '__main__':
    if project_root not in sys.path:
        sys.path.append(project_root)

import coloredlogs, logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

from src.models.model_config import model_cfg
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.m_utils.base_dataset import PreprocessedDataset
from src.models.estimate3d import MultiEstimator
from src.m_utils.evaluate import numpify
from src.m_utils.mem_dataset import MemDataset
from src.m_utils.tracker import Track
from src.m_utils.visualize import plotTracked3d


def export(model, loader, test_range, show=False):
    pose_list = list()
    for img_id, imgs in enumerate(tqdm(loader)):
        try:
            pass
        except Exception as e:
            pass

        info_dicts = numpify(imgs)
        frame = int(info_dicts[0]['image_path'][0].split('-')[1])
        model.dataset = MemDataset(info_dict=info_dicts, camera_parameter=camera_parameter, template_name='Unified')
        poses3d = model.estimate3d(0, frame, show=show)
        #poses3d, tracked_poses3d = model.estimate3d(0, show=show)

        # fig = plotTracked3d(tracked_poses3d)
        # fig.show()
        # plt.show()

        pose_list.append(poses3d)
        #pose_list.append(tracked_poses3d)

    surviving_tracks = []
    for track in model.tracks:
        if len(track) >= 5:
            surviving_tracks.append(track)

    print('\n[smoothing]')
    tracks = []
    for track in tqdm(surviving_tracks):
        track = Track.smoothing(track, sigma=2, interpolation_range=50, relevant_jids=range(0, 17))
        tracks.append(track)

    tracks_by_frame = {}
    pose_by_track_and_frame = {}
    for frame in test_range:
        assert frame not in tracks_by_frame
        tracks_by_frame[frame] = []
        for tid, track in enumerate(tracks):
            frames = track.frames
            poses = track.poses
            for i, t in enumerate(frames):
                if t > frame:
                    break
                elif t == frame:
                    tracks_by_frame[frame].append(tid)
                    pose_by_track_and_frame[tid, frame] = 1000 * np.asarray(poses[i])

    return pose_list, tracks_by_frame, pose_by_track_and_frame


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument('-d', nargs='+', dest='datasets', required=True,
                        choices=['Shelf', 'Campus', 'ultimatum1', 'Hexagonos'])
    parser.add_argument('-dumped', nargs='+', dest='dumped_dir', default=None)
    parser.add_argument("-range", nargs="+", dest='range', type=int)
    args = parser.parse_args()

    test_model = MultiEstimator(cfg=model_cfg)
    for dataset_idx, dataset_name in enumerate(args.datasets):
        model_cfg.testing_on = dataset_name
        if dataset_name == 'Shelf':
            dataset_path = model_cfg.shelf_path
            # you can change the test_rang to visualize different images (0~3199)
            test_range = range(605, 1800, 5)
            gt_path = dataset_path

        elif dataset_name == 'Campus':
            dataset_path = model_cfg.campus_path
            # you can change the test_rang to visualize different images (0~1999)
            test_range = [i for i in range(605, 1000, 5)]
            gt_path = dataset_path

        elif dataset_name == 'Hexagonos':
            dataset_path = model_cfg.hexagonos_path
            test_range = [i for i in range(args.range[0], args.range[1], 1)]
            gt_path = dataset_path

        else:
            logger.error(f"Unknown datasets name: {dataset_name}")
            exit(-1)

        # read the camera parameter of this dataset
        with open(osp.join(dataset_path, 'camera_parameter.pickle'), 'rb') as f:
            camera_parameter = pickle.load(f)

        # using preprocessed 2D poses
        test_dataset = PreprocessedDataset(args.dumped_dir[dataset_idx], test_range)
        logger.info(f"Using pre-processed datasets {args.dumped_dir[dataset_idx]} for quicker evaluation")

        test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=False, num_workers=6, shuffle=False)
        pose_in_range, tracks_by_frame, pose_by_track_and_frame = export(test_model, test_loader, test_range, show=False)
        test_range_str = '_' + str(args.range[0]) + '_' + str(args.range[1])
        os.makedirs(osp.join(model_cfg.root_dir, 'result'), exist_ok=True)
        with open(osp.join(model_cfg.root_dir, 'result', model_cfg.testing_on + '_tracks_by_frame' + test_range_str + '.pkl'), 'wb') as f:
            pickle.dump([tracks_by_frame, pose_by_track_and_frame], f)


