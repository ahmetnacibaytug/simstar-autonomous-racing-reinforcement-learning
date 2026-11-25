"""Summarize the stored evaluation CSVs without rerunning simulations."""
import csv
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
patterns = {
    'SAC': ('sac', 'test_*.csv'),
    'TD3': ('td3', 'test_*.csv'),
    'PPO': ('ppo', 'test_*.csv'),
}
for model, (folder, pattern) in patterns.items():
    for path in sorted((root / folder).glob(pattern)):
        with path.open(encoding='utf-8-sig', newline='') as source:
            rows = list(csv.DictReader(source))
        if not rows:
            continue
        lap = [float(row['lap_percent']) for row in rows]
        speed = [float(row['avg_speed_kmh']) for row in rows]
        steps = [int(row['steps']) for row in rows]
        print(f'{model:3} {path.name}: n={len(rows)}, lap={mean(lap):.1f}%, '
              f'speed={mean(speed):.1f} km/h, range={min(lap):.1f}–{max(lap):.1f}%, '
              f'episodes_at_20000={steps.count(20000)}')
