import re
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

num_re = re.compile(r'(-?\d+\.?\d*)')
sample_re = re.compile(r'^\s*(\d{3})\b')

# targeted regexes for the three fields
mean_re = re.compile(r'Mean\s*[:=]?\s*(-?\d+\.?\d*)', re.IGNORECASE)
std_re = re.compile(r'Std(?:\s*dev)?\s*[:=]?\s*(-?\d+\.?\d*)', re.IGNORECASE)
p20_re = re.compile(r'%\s*>\s*20%?\s*[=:]?\s*(-?\d+\.?\d*)', re.IGNORECASE)


def parse_quality_txt(path):
    """Parse the text file and return aggregated stats per category.

    Returns a dict mapping normalized category -> dict with lists for means, stds, pct20.
    """
    data = defaultdict(lambda: {"means": [], "stds": [], "p20": []})
    current_sample = None

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # detect sample header like: 000
            m = sample_re.match(line)
            if m and len(line) <= 10 and ':' not in line:
                current_sample = m.group(1)
                continue

            # expect lines with a category and numbers, e.g.
            # Translation u1: Mean=-4.78%, Std=2.81%, %>20%=0.08%
            if ':' in line:
                cat, rest = line.split(':', 1)
                key = cat.strip().lower().replace(' ', '_')

                # try targeted extraction first to avoid matching the literal '20' from '%>20'
                m_mean = mean_re.search(rest)
                m_std = std_re.search(rest)
                m_p20 = p20_re.search(rest)

                if m_mean and m_std and m_p20:
                    try:
                        mean_val = float(m_mean.group(1))
                        std_val = float(m_std.group(1))
                        p20_val = float(m_p20.group(1))
                    except ValueError:
                        continue
                else:
                    # fallback: extract numbers but ignore the isolated '20' token that comes from '%>20'
                    nums = [n for n in num_re.findall(rest) if n != '20']
                    if len(nums) < 3:
                        continue
                    try:
                        mean_val = float(nums[0])
                        std_val = float(nums[1])
                        p20_val = float(nums[2])
                    except ValueError:
                        continue

                data[key]['means'].append(mean_val)
                data[key]['stds'].append(std_val)
                data[key]['p20'].append(p20_val)

    return data


def aggregate(data):
    """Compute averages and counts for parsed data."""
    rows = []
    for key, vals in data.items():
        means = np.array(vals['means']) if vals['means'] else np.array([])
        stds = np.array(vals['stds']) if vals['stds'] else np.array([])
        p20s = np.array(vals['p20']) if vals['p20'] else np.array([])
        count = len(means)
        if count == 0:
            continue
        rows.append({
            'category': key,
            'count': int(count),
            'mean_of_means': float(np.nanmean(means)),
            'mean_of_stds': float(np.nanmean(stds)),
            'mean_of_%>20': float(np.nanmean(p20s))
        })
    return pd.DataFrame(rows).sort_values('category').reset_index(drop=True)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Parse quality text file and print global averages per category')
    p.add_argument('txtfile', nargs='?', default=None, help='path to the .txt file to parse (optional)')
    args = p.parse_args()

    # Determine which file to use: CLI arg takes precedence, otherwise look for common filenames
    if args.txtfile:
        txt_path = Path(args.txtfile)
        if not txt_path.exists():
            print(f"Provided file does not exist: {txt_path}")
            raise SystemExit(1)
    else:
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / 'Raw Speckle Quality.txt',
            script_dir / 'Raw_Speckle_Quality.txt',
            script_dir / 'raw_speckle_quality.txt'
        ]
        txt_path = next((p for p in candidates if p.exists()), None)
        if txt_path is None:
            print('No input file provided and no default file found in script directory.')
            print('Pass the path as an argument: python quality_data.py /path/to/file.txt')
            raise SystemExit(1)

    print(f"Using input file: {txt_path}")
    parsed = parse_quality_txt(str(txt_path))

    # categories to report (normalized keys used in parsing)
    report_cats = [
        ('Translation u1', 'translation_u1'),
        ('Translation u2', 'translation_u2'),
        ('Uniaxial u1', 'uniaxial_u1'),
        ('Uniaxial u2', 'uniaxial_u2'),
        ('Shear u1', 'shear_u1'),
        ('Shear u2', 'shear_u2'),
        ('Point Load u1', 'point_load_u1'),
        ('Point Load u2', 'point_load_u2'),
    ]

    print('\nGlobal averages (across all samples):')
    for display, key in report_cats:
        vals = parsed.get(key)
        if not vals or len(vals['means']) == 0:
            print(f"{display}: no data")
            continue
        means = np.array(vals['means'])
        stds = np.array(vals['stds'])
        p20s = np.array(vals['p20'])
        mm = np.nanmean(means)
        ms = np.nanmean(stds)
        mp = np.nanmean(p20s)
        print(f"{display}: Mean of means = {mm:.2f}%, Mean of stds = {ms:.2f}%, Mean of %>20 = {mp:.2f}%")

    print()
