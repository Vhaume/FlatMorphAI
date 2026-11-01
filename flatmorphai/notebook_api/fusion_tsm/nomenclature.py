"""nomenclature.py -- the frame-naming bridge between this project's training data and the host application.

WHY THIS FILE EXISTS
  The model is temporal: every prediction uses a 3-frame window (T-1, T0, T+1). To build that window the pipeline must
  know, for any labelled frame, which file is the previous frame and which is the next. Two naming conventions are in
  play, and they encode the time index differently, so this module maps both onto the SAME window rule.

THIS PROJECT (training data) convention
  Filenames are  <set>_Frame_<n>_in_<tot>.<ext>  where <n> is the 1-based frame index and <tot> the total number of
  frames in that video/worm clip. The window is chosen by dyn_window(n, tot) below: interior frames use [n-1, n, n+1],
  the first frame uses [1, 2, 3], the last uses [tot-2, tot-1, tot]. T0 is always the labelled frame, the two
  neighbours are the dynamic window. This is exactly what gen_temporal_dataset and precompute_mask3_metrics use.

HOST APPLICATION (FlatMorphAI) convention
  Frames are  <video_stem>_<second>s.<ext>  inside a directory named  <video_stem>_total_<total>s_step_<step>s . Frames
  are ordered by their <second> value, sampled every <step> seconds. There is no _Frame_n_in_tot token. See the
  FlatMorphAI naming module (flatmorphai/notebook_api/naming.py) for the full grammar (parse_image_name,
  parse_directory_name).

THE MAPPING (identical window, different index source)
  For the application, the frame's position n is its RANK in the clip ordered by second, and tot is the number of
  frames in the clip. Apply the same dyn_window(n, tot) to those ranks, then map the chosen ranks back to the
  application filenames. So  Frame_n_in_tot  is just the n-th ordered frame of the clip, and the temporal neighbours
  are the adjacent frames in acquisition order under either convention. temporal_window_app() does this for you and is
  the only function the application integration needs to call.

No third-party dependency beyond the standard library. random_state is irrelevant here (deterministic).
"""
import os
import re

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# --------------------------------------------------------------------------------------
# THIS PROJECT (training) convention:  <set>_Frame_<n>_in_<tot>.<ext>
# --------------------------------------------------------------------------------------
FRAME_RE = re.compile(r'^(?P<set>.+)_[Ff]rame_(?P<n>\d+)_in_(?P<tot>\d+)$')


def parse_frame(stem_or_name):
    """Return (set_id, n, tot) for a training-style name, or None."""
    m = FRAME_RE.match(os.path.splitext(os.path.basename(str(stem_or_name)))[0])
    return (m.group('set'), int(m.group('n')), int(m.group('tot'))) if m else None


def frame_name(set_id, k, tot, ext='.png'):
    """Build the training-style filename for index k."""
    return '%s_Frame_%d_in_%d%s' % (set_id, k, tot, ext)


def dyn_window(n, tot):
    """The 3-frame temporal window used everywhere in training. T0 is the labelled frame at index n."""
    if tot <= 1:
        return [n, n, n]
    if tot == 2:
        return [1, 2, 2] if n <= 1 else [1, 1, 2]
    if n <= 1:
        return [1, 2, 3]
    if n >= tot:
        return [tot - 2, tot - 1, tot]
    return [n - 1, n, n + 1]


def temporal_window_training(frame_name_or_path, ext='.png'):
    """Return the three training-style neighbour filenames (prev, cur, next) for a labelled frame."""
    pf = parse_frame(frame_name_or_path)
    if pf is None:
        return None
    sid, n, tot = pf
    ks = dyn_window(n, tot)
    return [frame_name(sid, k, tot, ext) for k in ks]


# --------------------------------------------------------------------------------------
# HOST APPLICATION convention:  <video_stem>_<second>s.<ext>  in  <stem>_total_<total>s_step_<step>s
# --------------------------------------------------------------------------------------
_APP_SECOND_RE = re.compile(r'_(?P<second>[0-9]+(?:\.[0-9]+)?)s?$')   # trailing _<second> with optional 's' unit marker


def app_second(name):
    """Extract the <second> value from an application frame filename, or None."""
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    m = _APP_SECOND_RE.search(stem)
    return float(m.group('second')) if m else None


def list_clip_frames(clip_dir):
    """Return the application frames of a clip directory, sorted by their <second> value: a list of (second, path)."""
    out = []
    for fn in os.listdir(clip_dir):
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        s = app_second(fn)
        if s is not None:
            out.append((s, os.path.join(clip_dir, fn)))
    out.sort(key=lambda t: t[0])
    return out


def temporal_window_app(frame_path, clip_dir=None):
    """Return (prev_path, cur_path, next_path) for an application frame, using the SAME dyn_window rule on the clip's
    acquisition order. clip_dir defaults to the frame's own directory. Falls back to a one-sided window at the clip
    boundaries exactly like training."""
    clip_dir = clip_dir or os.path.dirname(os.path.abspath(frame_path))
    frames = list_clip_frames(clip_dir)
    if not frames:
        return None
    paths = [p for _, p in frames]
    target = os.path.basename(frame_path)
    idx = next((i for i, p in enumerate(paths) if os.path.basename(p) == target), None)
    if idx is None:
        return None
    n = idx + 1                       # 1-based rank in acquisition order
    tot = len(paths)
    ks = dyn_window(n, tot)           # identical window rule
    return [paths[k - 1] for k in ks]


# --------------------------------------------------------------------------------------
# Unified entry point: detect the convention and return (prev, cur, next) absolute paths
# --------------------------------------------------------------------------------------
def temporal_window(frame_path, clip_dir=None):
    """Convention-agnostic. If the name matches the training pattern and clip_dir holds those files, use it, otherwise
    fall back to the application (ordered-by-second) rule. Returns three absolute paths (prev, cur, next)."""
    base = os.path.basename(frame_path)
    clip_dir = clip_dir or os.path.dirname(os.path.abspath(frame_path))
    if parse_frame(base) is not None:
        names = temporal_window_training(base, ext=os.path.splitext(base)[1] or '.png')
        cand = [os.path.join(clip_dir, n) for n in names]
        if all(os.path.exists(c) for c in cand):
            return cand
    return temporal_window_app(frame_path, clip_dir)
