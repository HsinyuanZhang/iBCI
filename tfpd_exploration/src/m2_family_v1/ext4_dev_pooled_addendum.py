"""One-time immutable pooled-metric addendum; replays exact existing ext4 IDs."""
from . import ext4_dev_replay as replay
replay.OUT = replay.ROOT / 'tfpd_exploration/results/m2/family_v1/ext4_e8_spint_dev_pooled_addendum_v1.json'
if __name__ == '__main__':
 import json
 print(json.dumps(replay.run(),sort_keys=True))
