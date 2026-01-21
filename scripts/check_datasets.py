"""Offline dataset checker (no network). Scans local folders for expected datasets and writes a report.
Usage: python scripts/check_datasets.py
"""
import os
import json
from dataset_utils import count_images_in_dir

CHECK_FOLDERS = {
    'PlantVillage': 'plantvillage',
    'Plant_Pathology': 'plant_pathology',
    'Plant_Seedlings': 'plant_seedlings',
    'Crop_Disease': 'crop_disease',
    'IP102': 'ip102',
    'PlantDoc': 'plantdoc',
    'Drought_Detection': 'drought',
    'Heat_Stress': 'heat_stress',
}

report = {}
for k, p in CHECK_FOLDERS.items():
    if os.path.exists(p):
        cnt = count_images_in_dir(p)
        report[k] = {'present': True, 'root': p, 'images': cnt}
    else:
        report[k] = {'present': False, 'root': p, 'images': 0}

with open('local_datasets_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print('Wrote local_datasets_report.json')
for k, v in report.items():
    print(f"{k}: present={v['present']}, images={v['images']}, root={v['root']}")