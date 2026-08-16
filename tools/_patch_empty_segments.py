from pathlib import Path

# 1) patch pressure_alignment loader
p = Path("utils/pressure_alignment.py")
text = p.read_text(encoding="utf-8")
old = '''    for segment in segments:
        left_path = _segment_csv_path(rec_dir, "left", segment)
        right_path = _segment_csv_path(rec_dir, "right", segment)
        if left_path is not None:
            t_arr, v_arr = _load_pressure_sensor_rows(left_path)
            left_times.append(t_arr)
            left_values.append(v_arr)
            left_source = left_path
        if right_path is not None:
            t_arr, v_arr = _load_pressure_sensor_rows(right_path)
            right_times.append(t_arr)
            right_values.append(v_arr)
            right_source = right_path
'''
new = '''    for segment in segments:
        left_path = _segment_csv_path(rec_dir, "left", segment)
        right_path = _segment_csv_path(rec_dir, "right", segment)
        if left_path is not None:
            try:
                t_arr, v_arr = _load_pressure_sensor_rows(left_path)
            except ValueError as exc:
                # Some reconstructed segments are header-only / zero-row.
                if "empty" in str(exc).lower() or "no numeric rows" in str(exc).lower():
                    pass
                else:
                    raise
            else:
                if len(t_arr) > 0:
                    left_times.append(t_arr)
                    left_values.append(v_arr)
                    left_source = left_path
        if right_path is not None:
            try:
                t_arr, v_arr = _load_pressure_sensor_rows(right_path)
            except ValueError as exc:
                if "empty" in str(exc).lower() or "no numeric rows" in str(exc).lower():
                    pass
                else:
                    raise
            else:
                if len(t_arr) > 0:
                    right_times.append(t_arr)
                    right_values.append(v_arr)
                    right_source = right_path
'''
if old not in text:
    raise SystemExit('loader loop not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')
print('patched pressure_alignment loader')

# 2) also make _load_pressure_sensor_rows tolerate header-only more clearly? already raises empty.
print('done')
