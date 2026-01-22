from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import pandas as pd
import random

def generate_high_contrast_data(labels, n_per_class=100, mode='train'):
    texts, images = [], []
    signatures = {
        'water_stress': {'color': (139, 69, 19), 'shape': 'line'},
        'nutrient_def': {'color': (255, 255, 0), 'shape': 'circle'},
        'pest_risk': {'color': (0, 0, 0), 'shape': 'dots'},
        'disease_risk': {'color': (128, 0, 128), 'shape': 'blob'},
        'heat_stress': {'color': (255, 69, 0), 'shape': 'border'}
    }

    for label in labels:
        sig = signatures[label]
        for i in range(n_per_class):
            t = f"Report: {label} detected. "
            if label == 'water_stress': t += "Soil is dry, cracking, wilting leaves."
            elif label == 'nutrient_def': t += "Chlorosis, yellowing veins, pale leaves."
            elif label == 'pest_risk': t += "Larvae found, bite marks, holes in leaves."
            elif label == 'disease_risk': t += "Fungal spores, necrotic lesions, rotting."
            elif label == 'heat_stress': t += "Sun scald, burnt tips, drooping."

            im = Image.new('RGB', (224, 224), (34, 139, 34))
            draw = ImageDraw.Draw(im)
            col = sig['color']
            if sig['shape'] == 'line':
                for _ in range(5):
                    draw.line((random.randint(0,224), 0, random.randint(0,224), 224), fill=col, width=5)
            elif sig['shape'] == 'circle':
                draw.ellipse((50,50,174,174), fill=col)
            elif sig['shape'] == 'dots':
                for _ in range(30):
                    x, y = random.randint(10,210), random.randint(10,210)
                    draw.ellipse((x, y, x+10, y+10), fill=col)
            elif sig['shape'] == 'blob':
                draw.rectangle((60,60,160,160), fill=col)
            elif sig['shape'] == 'border':
                draw.rectangle((0,0,224,224), outline=col, width=20)

            if mode == 'val':
                im = im.filter(ImageFilter.GaussianBlur(1))

            # Use consistent column name `labels` (list of ints) to match the rest of the
            # codebase which expects `labels` to be a list per-example (e.g. [2]).
            texts.append({'text': t, 'labels': [labels.index(label)], 'label_name': label, 'modality': 'text'})
            images.append({'image': im, 'labels': [labels.index(label)], 'label': labels.index(label), 'modality': 'image'})

    return pd.DataFrame(texts), pd.DataFrame(images)
