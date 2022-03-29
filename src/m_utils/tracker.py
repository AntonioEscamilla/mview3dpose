import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


class Track:
    def __init__(self, t, pose, feats=None, last_seen_delay=5):
        """
        :param t: init time
        :param pose: ndarray(3, 17)
        :param last_seen_delay: max delay between times
        """
        self.frames = [int(t)]
        if type(pose) == list:
            pose = pose
        else:
            pose = list(pose.T)                             # a pose is ndarray(3,17) and we need a list of array(3,)

        self.J = len(pose)
        self.poses = [pose]
        self.reid_feats = feats
        self.last_seen_delay = last_seen_delay
        self.lookup = None

    def __len__(self):
        if len(self.frames) == 1:
            return 1
        else:
            first = self.frames[0]
            last = self.frames[-1]
            return last - first + 1

    def last_seen(self):
        return self.frames[-1]

    def first_frame(self):
        return self.frames[0]

    def add_pose(self, t, pose, feats=None):
        last_t = self.last_seen()
        assert last_t < t
        diff = t - last_t
        assert diff <= self.last_seen_delay

        self.frames.append(t)
        if type(pose) == list:
            self.poses.append(pose)
        else:
            self.poses.append(list(pose.T))                 # a pose is ndarray(3,17) and we need a list of array(3,)
        self.reid_feats = feats
        self.lookup = None                                  # reset lookup

    def get_by_frame(self, t):
        if self.lookup is None:
            self.lookup = {}
            for f, pose in zip(self.frames, self.poses):
                self.lookup[f] = pose

        if t in self.lookup:
            return self.lookup[t]
        else:
            return None

    @staticmethod
    def smoothing(track, sigma,
                  interpolation_range=4,
                  relevant_jids=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]):
        """ smoothing of a track
        :param track:
        :param sigma:
        :param interpolation_range:
        :param relevant_jids: is set up for mscoco
        :return:
        """
        first_frame = track.first_frame()
        last_frame = track.last_seen() + 1
        n_frames = last_frame - first_frame

        relevant_jids_lookup = {}
        relevant_jids = set(relevant_jids)

        delete_jids = []

        # step 0: make sure all relevent jids have entries
        for jid in relevant_jids:
            jid_found = False
            for frame in range(first_frame, last_frame):
                pose = track.get_by_frame(frame)
                if pose is not None and pose[jid] is not None:
                    jid_found = True
                    break

            if not jid_found:
                delete_jids.append(jid)

        for jid in delete_jids:
            relevant_jids.remove(jid)

        # step 1:
        unrecoverable = set()
        for jid in relevant_jids:
            XYZ = np.empty((n_frames, 3))
            for frame in range(first_frame, last_frame):
                pose = track.get_by_frame(frame)

                if pose is None or pose[jid] is None:
                    start_frame = max(first_frame, frame - interpolation_range)
                    end_frame = min(last_frame, frame + interpolation_range)

                    from_left = []
                    for _frame in range(start_frame, frame):
                        _pose = track.get_by_frame(_frame)
                        if _pose is None or _pose[jid] is None:
                            continue
                        from_left.append(_pose[jid])

                    from_right = []
                    for _frame in range(frame, end_frame):
                        _pose = track.get_by_frame(_frame)
                        if _pose is None or _pose[jid] is None:
                            continue
                        from_right.append(_pose[jid])

                    pts = []
                    if len(from_left) > 0:
                        pts.append(from_left[-1])
                    if len(from_right) > 0:
                        pts.append(from_right[0])

                    if len(pts) > 0:
                        pt = np.mean(pts, axis=0)
                    else:
                        unrecoverable.add((jid, frame))
                        pt = np.array([0., 0., 0.])

                else:
                    pt = pose[jid]
                XYZ[frame - first_frame] = pt

            XYZ_sm = np.empty_like(XYZ)
            for dim in [0, 1, 2]:
                D = XYZ[:, dim]
                D = gaussian_filter1d(D, sigma, mode='reflect')
                XYZ_sm[:, dim] = D
            relevant_jids_lookup[jid] = XYZ_sm

        new_track = None

        for frame in range(first_frame, last_frame):
            person = []
            for jid in range(track.J):
                if jid in relevant_jids_lookup:
                    if (jid, frame) in unrecoverable:
                        person.append(None)
                    else:
                        XYZ_sm = relevant_jids_lookup[jid]
                        pt = XYZ_sm[frame - first_frame]
                        person.append(pt)
                else:
                    pose = track.get_by_frame(frame)
                    if pose is None:
                        person.append(None)
                    else:
                        person.append(pose[jid])
            if new_track is None:
                new_track = Track(frame, person, last_seen_delay=track.last_seen_delay)
            else:
                new_track.add_pose(frame, person)

        return new_track
