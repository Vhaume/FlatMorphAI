"""golden_targets.py -- the single source of truth for the GOLDEN frame set.

Imported by golden_eval.py (evaluation) and precompute_mask3_metrics.py (metric generation), so the metrics written to
analysis_out/golden_metrics.csv cover exactly the frames golden_eval scores, with no drift between the two. Each EXPECT
entry is (group label, keyword filters that must all appear in the file name, allowed frame indices, expected class).
"""
import os, re

U = 'µ'
EXPECT = [
 ('Amitraze control', ['Amitraze', 'R2(5)', '240523', '_T0', '_0' + U + 'M'], list(range(16, 31)), 'Wildtype'),
 ('GABA screw', ['GABA', '10' + U + 'M', '_T1h', 'R3(2)', '210323'], list(range(5, 17)), 'Screw-like'),
 ('Amitraze_0uM R3(1) f7', ['Amitraze', '0,5%DMSO', '_T0_', 'R3(1)', '250523'], [7], 'Screw-like'),
 ('Amitraze_15uM R3(4) f2', ['Amitraze', '15' + U + 'M', '_T0_', 'R3(4)', '250523'], [2], 'Screw-like'),
 ('Glutamate_0uM R2(1) f25', ['Glutamate', '_0' + U + 'M', '_T3h', 'R2(1)', '130323'], [25], 'Screw-like'),
 ('Glutamate_10000uM R3(5) f10', ['Glutamate', '10000' + U + 'M', '_T0', 'R3(5)', '140323'], [10], 'Screw-like'),
 ('Haloperidol_10uM R1(2) f1', ['Haloperidol', '10' + U + 'M', '_T0', 'R1(2)', '230523'], [1], 'Screw-like'),
 ('Haloperidol_10uM R3(2) f6', ['Haloperidol', '10' + U + 'M', '_T0', 'R3(2)', '250523'], [6], 'Screw-like'),
 ('Baclofen_0uM R3(3) f2', ['Baclofen', '_0' + U + 'M', '_T3h', 'R3(3)', '050623'], [2], 'Screw-like'),
]
def _n(t): return str(t).lower().replace('μ', U)
def _has(nm, keys): return all(_n(k) in _n(nm) for k in keys)
FRAME_RE = re.compile(r'^(?P<set>.+)_[Ff]rame_(?P<n>\d+)_in_(?P<tot>\d+)$')
def parse(s):
    m = FRAME_RE.match(os.path.splitext(os.path.basename(s))[0]); return (m.group('set'), int(m.group('n')), int(m.group('tot'))) if m else None
def set_id(s): return re.sub(r'_[Ff]rame_\d+_in_\d+.*$', '', os.path.basename(str(s)))
def _fname(s, n, tot): return '%s_Frame_%d_in_%d.png' % (s, n, tot)
