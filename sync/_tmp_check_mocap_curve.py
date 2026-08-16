from pathlib import Path
import tempfile, textwrap
import numpy as np
from utils.bvh_parser import load_bvh_motion_preserve_frames
from utils.pressure_alignment import load_mocap_foot_curves, _joint_vertical_axis
from utils.bvh_pose import compute_joint_positions, transform_display_positions

frames = []
for i in range(10):
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
path = root / 'test.bvh'
path.write_text(bvh, encoding='utf-8')
motion = load_bvh_motion_preserve_frames(path)

for preset in ['zup','raw']:
    vaxis = _joint_vertical_axis(preset)
    heights_l=[]
    for fi in range(len(motion.raw_frames)):
        pos = transform_display_positions(compute_joint_positions(motion, fi), preset)
        idx = [i for i,j in enumerate(motion.joints) if 'left' in j.name.lower() and 'foot' in j.name.lower()]
        heights_l.append(float(np.min(pos[idx, vaxis])))
    curves = load_mocap_foot_curves(motion, axis_preset=preset)
    print('preset', preset, 'vaxis', vaxis)
    print(' heights_l', heights_l)
    print(' curve_l  ', [round(x,3) for x in curves.left_sum.tolist()])
    print(' curve_r  ', [round(x,3) for x in curves.right_sum.tolist()])
    print(' ground mean L', float(np.mean(curves.left_sum[:5])), 'lift mean L', float(np.mean(curves.left_sum[5:])))
    print('---')
