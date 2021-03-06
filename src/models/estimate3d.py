import sys
import os.path as osp

# Config project if not exist
project_path = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src.models.model_config import model_cfg
import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

import cv2
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.m_utils.geometry import geometry_affinity, get_min_reprojection_error, check_bone_length, bundle_adjustment, \
    multiTriIter
from backend.CamStyle.feature_extract import FeatureExtractor, pairwise_distance
# from backend.reid.torchreid.utils.feature_extractor import FeatureExtractor, pairwise_distance
from src.models.matchSVT import matchSVT
from src.m_utils.visualize import show_panel_mem, plotPaperRows
from collections import OrderedDict
from src.m_utils.tracker import Track
from src.models import pictorial

# from src.m_lib import pictorial


sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89],
                  dtype=np.float32) / 10.0
vars_ = (sigmas * 2) ** 2


class MultiEstimator(object):
    def __init__(self, cfg, debug=False):
        self.extractor = FeatureExtractor()
        # self.extractor = FeatureExtractor(
        #     model_name='osnet_x1_0',
        #     model_path='c:\\Python Projects\\MultiView_MultiPeople_Pose\\05_mview3dpose\\backend\\reid\\models\\osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        #     device='cuda'
        # )
        self.cfg = cfg
        self.dataset = None
        self.tracks = []

    def estimate3d(self, img_id, frame, show=False, plt_id=0, last_seen_delay=14):
        data_batch = self.dataset[img_id]
        affinity_mat, query_features, query = self.extractor.get_affinity(data_batch, rerank=self.cfg.rerank)
        if self.cfg.rerank:
            affinity_mat = torch.from_numpy(affinity_mat)
            affinity_mat = torch.max(affinity_mat, affinity_mat.t())
            affinity_mat = 1 - affinity_mat
        else:
            affinity_mat = affinity_mat.cpu()
        dimGroup = self.dataset.dimGroup[img_id]

        info_list = list()
        for cam_id in self.dataset.cam_names:
            info_list += self.dataset.info_dict[cam_id][img_id]

        pose_mat = np.array([i['pose2d'] for i in info_list]).reshape(-1, model_cfg.joint_num, 3)[..., :2]
        geo_affinity_mat = geometry_affinity(pose_mat.copy(), self.dataset.F.numpy(), self.dataset.dimGroup[img_id])
        geo_affinity_mat = torch.tensor(geo_affinity_mat)
        if self.cfg.metric == 'geometry mean':
            W = torch.sqrt(affinity_mat * geo_affinity_mat)
        elif self.cfg.metric == 'circle':
            W = torch.sqrt((affinity_mat ** 2 + geo_affinity_mat ** 2) / 2)
        elif self.cfg.metric == 'Geometry only':
            W = torch.tensor(geo_affinity_mat)
        elif self.cfg.metric == 'ReID only':
            W = torch.tensor(affinity_mat)
        else:
            logger.critical('Get into default option, are you intend to do it?')
            _alpha = 0.8
            W = 0.8 * affinity_mat + (1 - _alpha) * geo_affinity_mat
        W[torch.isnan(W)] = 0  # Some times (Shelf 452th img eg.) torch.sqrt will return nan if its too small
        sub_imgid2cam = np.zeros(pose_mat.shape[0], dtype=np.int32)
        for idx, i in enumerate(range(len(dimGroup) - 1)):
            sub_imgid2cam[dimGroup[i]:dimGroup[i + 1]] = idx

        num_person = 10
        X0 = torch.rand(W.shape[0], num_person)

        # Use spectral method to initialize assignment matrix.
        if self.cfg.spectral:
            eig_value, eig_vector = W.eig(eigenvectors=True)
            _, eig_idx = torch.sort(eig_value[:, 0], descending=True)

            if W.shape[1] >= num_person:
                X0 = eig_vector[eig_idx[:num_person]].t()
            else:
                X0[:, :W.shape[1]] = eig_vector.t()

        match_mat = matchSVT(W, dimGroup, alpha=self.cfg.alpha_SVT, _lambda=self.cfg.lambda_SVT,
                             dual_stochastic_SVT=self.cfg.dual_stochastic_SVT)

        bin_match = match_mat[:, torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).squeeze()] > 0.9
        bin_match = bin_match.reshape(W.shape[0], -1)

        matched_list = [[] for i in range(bin_match.shape[1])]
        for sub_imgid, row in enumerate(bin_match):
            if row.sum() != 0:
                pid = row.argmax()
                matched_list[pid].append(sub_imgid)

        matched_list = [np.array(i) for i in matched_list]
        if self.cfg.hybrid:
            multi_pose3d = self._hybrid_kernel(matched_list, pose_mat, sub_imgid2cam, img_id)
            chosen_img = [[]] * len(sub_imgid2cam)
        else:
            multi_pose3d, chosen_img = self._top_down_pose_kernel(geo_affinity_mat, matched_list, pose_mat,
                                                                  sub_imgid2cam)
        if show:  # hybrid not implemented yet.
            bin_match = match_mat[:, torch.nonzero(torch.sum(match_mat, dim=0) > 0.9).squeeze()] > 0.9
            bin_match = bin_match.reshape(W.shape[0], -1)
            matched_list = [[] for i in range(bin_match.shape[1])]
            for sub_imgid, row in enumerate(bin_match):
                if row.sum() != 0:
                    pid = row.argmax()
                    matched_list[pid].append(sub_imgid)
            matched_list = [np.array(i) for i in matched_list]
            show_panel_mem(self.dataset, matched_list, info_list, sub_imgid2cam, img_id, affinity_mat,
                           geo_affinity_mat, W, plt_id, multi_pose3d)
            plotPaperRows(self.dataset, matched_list, info_list, sub_imgid2cam, img_id, affinity_mat,
                          geo_affinity_mat, W, plt_id, multi_pose3d)

        candidates = []
        for chosen_idx in chosen_img:
            if len(chosen_idx) > 1:
                matched_query = [query[i] for i in chosen_idx]
                matched_features = OrderedDict((key, query_features[key]) for key, _, _ in matched_query)
                candidates.append([matched_features, matched_query])

        # ------------------------------------
        # Eliminate duplicated candidate poses
        # ------------------------------------
        distances = []  # (hid1, hid2, distance)
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                feat_distance = pairwise_distance(candidates[i][0], candidates[j][0], candidates[i][1], candidates[j][1])
                feat_distance = feat_distance.min().item()
                pose_similarity = get_similarity(multi_pose3d[i], multi_pose3d[j])
                distances.append((i, j, feat_distance, pose_similarity))
        mergers_root = {}  # hid -> root
        mergers = {}  # root: [ hid, hid, .. ]
        all_merged_hids = set()
        for hid1, hid2, distance, similarity in distances:
            # if distance > 0.24:  # distance > 0.18 for OsNet reid
            if distance > 0.2 and similarity < 16:     # 15
                continue

            # if distance > 0.33:
            #     continue

            if hid1 in mergers_root and hid2 in mergers_root:
                continue  # both are already handled

            if hid1 in mergers_root:
                hid1 = mergers_root[hid1]

            if hid1 not in mergers:
                mergers[hid1] = [hid1]

            mergers[hid1].append(hid2)
            mergers_root[hid2] = hid1
            all_merged_hids.add(hid1)
            all_merged_hids.add(hid2)

        merged_poses = []
        merged_candidates = []
        for hid in range(n):
            if hid in mergers:
                poses_list = [multi_pose3d[hid2] for hid2 in mergers[hid]]
                merged_poses.append(get_avg_pose(poses_list))
                merged_candidates.append(merge_candidates_feats(candidates, mergers[hid]))
            elif hid not in all_merged_hids:
                merged_poses.append(multi_pose3d[hid])
                merged_candidates.append(candidates[hid])

        multi_pose3d = merged_poses
        candidates = merged_candidates

        # ------------------------------------
        # Track poses
        # ------------------------------------
        possible_tracks = []
        for track in self.tracks:
            if track.last_seen() + last_seen_delay < frame:
                continue
            possible_tracks.append(track)

        n = len(possible_tracks)
        if n > 0:
            m = len(candidates)
            D = np.empty((n, m))
            for tid, track in enumerate(possible_tracks):
                for cid, candidate in enumerate(candidates):
                    between_frames_aff = pairwise_distance(track.reid_feats[0], candidate[0], track.reid_feats[1],
                                                           candidate[1]).cpu().detach().numpy()
                    D[tid, cid] = between_frames_aff.min()

            rows, cols = linear_sum_assignment(D)

            handled_pids = set()
            for tid, cid in zip(rows, cols):
                d = D[tid, cid]
                if d > 0.23:            # d > 0.21
                    continue

                # merge pose into track
                track = possible_tracks[tid]
                pose = multi_pose3d[cid]
                track.add_pose(frame, pose, candidates[cid])
                handled_pids.add(cid)

            # add all remaining poses as tracks
            for cid, candidate in enumerate(candidates):
                if cid in handled_pids:
                    continue
                track = Track(frame, multi_pose3d[cid], candidates[cid], last_seen_delay)
                self.tracks.append(track)

        else:
            for pid, pose in enumerate(multi_pose3d):
                track = Track(frame, pose, candidates[pid])
                self.tracks.append(track)

        return multi_pose3d

    def _hybrid_kernel(self, matched_list, pose_mat, sub_imgid2cam, img_id):
        # return pictorial.hybrid_kernel(self, matched_list, pose_mat, sub_imgid2cam, img_id)
        multi_pose3d = list()

        for person in matched_list:
            # use bottom-up approach to get the 3D pose of person
            if person.shape[0] <= 1:
                continue

            # step1: use the 2D joint of person to triangulate the 3D joints candidates
            # person's 17 3D joints candidates
            candidates = np.zeros((17, person.shape[0] * (person.shape[0] - 1) // 2, 3))  # 17xC^2_nx3

            cnt = 0
            for i in range(person.shape[0]):
                for j in range(i + 1, person.shape[0]):
                    cam_id_i, cam_id_j = sub_imgid2cam[person[i]], sub_imgid2cam[person[j]]
                    projmat_i, projmat_j = self.dataset.P[cam_id_i], self.dataset.P[cam_id_j]
                    pose2d_i, pose2d_j = pose_mat[person[i]].T, pose_mat[person[j]].T
                    pose3d_homo = cv2.triangulatePoints(projmat_i, projmat_j, pose2d_i, pose2d_j)
                    pose3d_ij = pose3d_homo[:3] / pose3d_homo[3]
                    candidates[:, cnt] += pose3d_ij.T
                    cnt += 1

            unary = self.dataset.get_unary(person, sub_imgid2cam, candidates, img_id)

            # step2: use the max-product algorithm to inference to get the 3d joint of the person
            # change the coco order
            coco_2_skel = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            candidates = np.array(candidates)[coco_2_skel]
            unary = unary[coco_2_skel]
            skel = pictorial.getskel()
            # construct pictorial model
            edges = pictorial.getPictoStruct(skel, self.dataset.distribution)
            xp = pictorial.inferPict3D_MaxProd(unary, edges, candidates)
            human = np.array([candidates[i][j] for i, j in zip(range(xp.shape[0]), xp)])
            human_coco = np.zeros((17, 3))
            human_coco[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = human
            human_coco[[1, 2, 3, 4]] = human_coco[0]  # Just make visualize beauty not real ear and eye
            human_coco = human_coco.T
            if self.cfg.reprojection_refine and len(person) > 2:
                for joint_idx in range(human_coco.shape[1]):
                    reprojected_error = np.zeros(len(person))
                    for idx, pid in enumerate(person):
                        human_coco_homo = np.ones(4)
                        human_coco_homo[:3] = human_coco[:, joint_idx]
                        projected_pose_homo = self.dataset.P[sub_imgid2cam[pid]] @ human_coco_homo
                        projected_pose = projected_pose_homo[:2] / projected_pose_homo[2]
                        reprojected_error[idx] += np.linalg.norm(projected_pose - pose_mat[pid, joint_idx])

                    pose_select = (
                                              reprojected_error - reprojected_error.mean()) / reprojected_error.std() < self.cfg.refine_threshold
                    if pose_select.sum() >= 2:
                        Ps = list()
                        Ys = list()
                        for idx, is_selected in enumerate(pose_select):
                            if is_selected:
                                Ps.append(self.dataset.P[sub_imgid2cam[person[idx]]])
                                Ys.append(pose_mat[person[idx], joint_idx].reshape(-1, 1))
                        Ps = torch.tensor(Ps, dtype=torch.float32)
                        Ys = torch.tensor(Ys, dtype=torch.float32)
                        Xs = multiTriIter(Ps, Ys)
                        refined_pose = (Xs[:3] / Xs[3]).numpy()
                        human_coco[:, joint_idx] = refined_pose.reshape(-1)
            if True or check_bone_length(human_coco):
                multi_pose3d.append(human_coco)
        return multi_pose3d

    def _top_down_pose_kernel(self, geo_affinity_mat, matched_list, pose_mat, sub_imgid2cam):
        multi_pose3d = list()
        chosen_img = list()
        for person in matched_list:
            Graph = geo_affinity_mat[person][:, person].clone().numpy()
            Graph *= (1 - np.eye(Graph.shape[0]))  # make diagonal 0
            if len(Graph) < 2:
                continue
            elif len(Graph) > 2:
                if self.cfg.use_mincut:
                    cut0, cut1 = find_mincut(Graph.copy())
                    cut = cut0 if len(cut0) > len(cut1) else cut1
                    cut = cut.astype(int)
                    sub_imageid = person[cut]
                else:
                    sub_imageid = get_min_reprojection_error(person, self.dataset, pose_mat, sub_imgid2cam)
            else:
                sub_imageid = person

            _, rank = torch.sort(geo_affinity_mat[sub_imageid][:, sub_imageid].sum(dim=0))
            sub_imageid = sub_imageid[rank[:2]]
            cam_id_0, cam_id_1 = sub_imgid2cam[sub_imageid[0]], sub_imgid2cam[sub_imageid[1]]
            projmat_0, projmat_1 = self.dataset.P[cam_id_0], self.dataset.P[cam_id_1]
            pose2d_0, pose2d_1 = pose_mat[sub_imageid[0]].T, pose_mat[sub_imageid[1]].T
            pose3d_homo = cv2.triangulatePoints(projmat_0, projmat_1, pose2d_0, pose2d_1)
            if self.cfg.use_bundle:
                pose3d_homo = bundle_adjustment(pose3d_homo, person, self.dataset, pose_mat, sub_imgid2cam,
                                                logging=logger)
            pose3d = pose3d_homo[:3] / (pose3d_homo[3] + 10e-6)
            # pose3d -= ((pose3d[:, 11] + pose3d[:, 12]) / 2).reshape ( 3, -1 ) # No need to normalize to hip
            if check_bone_length(pose3d):
                multi_pose3d.append(pose3d)
            else:
                # logging.info ( f'A pose proposal deleted on {img_id}:{person}' )
                sub_imageid = list()
                pass
            chosen_img.append(sub_imageid)
        return multi_pose3d, chosen_img


def get_avg_pose(poses):
    J = len(poses[0])
    result = [None] * J

    for jid in range(J):
        valid_points = []
        for pose in poses:
            if pose[jid] is not None:
                valid_points.append(pose[jid])
        if len(valid_points) > 0:
            result[jid] = np.mean(valid_points, axis=0)
        else:
            result[jid] = None
    return np.asarray(result)


def merge_candidates_feats(candidates, idxs):
    query_ = candidates[idxs[0]][1] + candidates[idxs[1]][1]
    feats_ = {**candidates[idxs[0]][0], **candidates[idxs[1]][0]}
    return [feats_, query_]


def get_similarity(pose1, pose2, threshold=0.80, num_kpts=17):
    num_similar_kpt = 0
    pose1, pose2 = 1000*pose1.T, 1000*pose2.T
    for kpt_id in range(num_kpts):
        distance = np.sum((pose1[kpt_id] - pose2[kpt_id]) ** 2)
        area = max(get_volume(pose1), get_volume(pose2))
        similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * vars_[kpt_id]))
        if similarity > threshold:
            num_similar_kpt += 1
    return num_similar_kpt


def get_bbox(keypoints):
    itmax = np.amax(keypoints, axis=0)
    itmin = np.amin(keypoints, axis=0)
    return [itmin[i] for i in range(3)] + [itmax[i] for i in range(3)]


def get_volume(keypoints):
    bbox = get_bbox(keypoints)
    volume = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2])
    return volume


def get_area(keypoints):
    bbox = get_bbox(keypoints)
    max_x_y = max(bbox[3] - bbox[0], bbox[4] - bbox[1])
    volume = max_x_y * (bbox[5] - bbox[2])
    return volume
