
import torch
import numpy as np


grading_files = ["outputs/statistics/layer_gradings/style_analysis_geometry_wolf",
                 "outputs/statistics/layer_gradings/style_analysis_geometry_cat",
                 "outputs/statistics/layer_gradings/style_analysis_geometry_shark",
                 "outputs/statistics/layer_gradings/style_analysis_geometry_horse",
                 "outputs/statistics/layer_gradings/style_analysis_geometry_cow"]

stats = [torch.load(f)['stats'] for f in grading_files]
layer_names = torch.load(grading_files[0])['layer_names']

combined = {}
for t in stats[0].keys():
    combined[t]  = {}
    for k in ['k', 'q']:
        combined[t][k] = []
        for l in range(70):
            layer_gradings = sorted([stat[t][k][l] for stat in stats])
            combined[t][k].append(np.mean(layer_gradings[1:-1]))

grading_file =  {'stats': combined, 'layer_names': layer_names}
torch.save(grading_file, "outputs/statistics/layer_gradings/combined_gradings/combined_layer_grading")