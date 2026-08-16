from pathlib import Path
import textwrap
import numpy as np
from utils.bvh_parser import load_bvh_motion_preserve_frames
from utils.pressure_alignment import load_mocap_foot_curves, _joint_vertical_axis, _candidate_joint_indices
from utils.bvh_pose import compute_joint_positions, transform_display_positions

frames = []
for i in range(10):
    # Y-up BVH: foot height is Y channel
    ly = 0.0 if i < 5 else 1.0
    ry = 0.0 if i >= 5 else 1.0
    frames.append(f"0 0 0 0 0 0 0 {ly} 0 0 {ry} 0")

bvh = textwrap.dedent('''
HIERARCHY
ROOT Hips
{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT LeftFoot
    {
        OFFSET 0 0 0
        CHANNELS 3 Xposition Yposition Zposition
        End Site { OFFSET 0 0 0 }
    }
    JOINT RightFoot
    {
        OFFSET 0 0 0
        CHANNELS 3 Xposition Yposition Zposition
        End Site { OFFSET 0 0 0 }
    }
}
MOTION
Frames: 10
Frame Time: 0.1
''').strip() + "\n" + "\n".join(frames)

root = Path('sync') / '_tmp_curve_check'
root.mkdir(parents=True, exist_ok=True)
path = root / 'test_yup.bvh'
path.write_text(bvh, encoding='utf-8')
motion = load_bvh_motion_preserve_frames(path)
print('joints', [j.name for j in motion.joints])

for preset in ['zup','raw']:
    print('===', preset, '===')
    vaxis_code = _joint_vertical_axis(preset)
    print('code vaxis', vaxis_code)
    for fi in [0,5]:
        raw = compute_joint_positions(motion, fi)
        disp = transform_display_positions(raw, preset)
        li = _candidate_joint_indices(motion, 'left')
        print('frame', fi, 'raw', raw[li], 'disp', disp[li], 'min_disp_axis_code', np.min(disp[li, vaxis_code]))
        # also report min along each axis
        print('  min axes raw', raw[li].min(axis=0), 'disp', disp[li].min(axis=0))
    curves = load_mocap_foot_curves(motion, axis_preset=preset)
    print('curve L', [round(x,3) for x in curves.left_sum.tolist()])
    print('curve R', [round(x,3) for x in curves.right_sum.tolist()])
