import os
import torch
import math
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

MEAN_IDX = 0
STD_IDX = 1

SERIES_NAME = 'style_analysis_texture_rabbit'

def compute_jsd(m1, s1, m2, s2):
    # Mean and variance of the intermediate distribution
    mM = 0.5 * (m1 + m2)
    sM2 = 0.5 * (s1**2 + s2**2)
    
    # Compute KL divergence terms
    kl1 = 0.5 * (np.sum(s1**2 / sM2) + np.sum((mM - m1)**2 / sM2) - len(m1) + np.sum(np.log(sM2 / s1**2)))
    kl2 = 0.5 * (np.sum(s2**2 / sM2) + np.sum((mM - m2)**2 / sM2) - len(m2) + np.sum(np.log(sM2 / s2**2)))
    
    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl1 + kl2)
    return jsd

def main(args):
    input_dir = f"outputs/statistics/image_series/{args.series_name}"
    layer_names = torch.load(os.path.join(input_dir, "layer_names"))
    stat_files = os.listdir(os.path.join(input_dir, "stats"))

    num_styles = len(set([file.split("_")[1] for file in stat_files]))
    num_samples = len(set([file.split("_")[2] for file in stat_files]))

    # stats = [[torch.load(os.path.join(input_dir, "stats", f"stats_{i}_{j}")) for j in range(num_samples)] for i in range(num_styles)]
    timesteps = list(torch.load(os.path.join(input_dir, "stats", "stats_0_0")).keys())

    grading_keys = ["q", "k", "v"]
    # grading = {k: {t: [] for t in timesteps} for k in grading_keys}
    grading = {t: {k: [] for k in grading_keys} for t in timesteps}


    similiar_style_combs = math.comb(num_samples, 2) * num_styles
    different_style_combs = math.comb(num_styles, 2) * num_samples**2

    mean_q, mean_k, mean_v, std_q, std_k, std_v = [], [], [], [], [], []
    for t in tqdm(timesteps):
        stats_t = [torch.load(os.path.join(input_dir, "stats", f"stats_{i}_{j}"))[t] for i in range(num_styles) for j in range(num_samples)]
        # stats_t = [stats[i][j][t] for i in range(len(stats)) for j in range(len(stats[i]))]  # get statistics of current time step only for all images
        sum_t = 0
        for li in range(len(layer_names)):  # calculate for each layer
            for st in grading_keys:  # choose stats type: queries/keys
                similarity_dists = 0
                difference_dists = 0
                for si in range(num_styles):
                    curr_style_indices = np.array(range(num_samples)) + si * 2
                    for im1_idx in curr_style_indices:
                        for im2_idx in range(im1_idx+1, (len(stats_t))):
                            m1, m2 = np.array(stats_t[im1_idx][f"stats_{st}"][li][MEAN_IDX].cpu()), np.array(stats_t[im2_idx][f"stats_{st}"][li][MEAN_IDX].cpu())
                            s1, s2 = np.array(stats_t[im1_idx][f"stats_{st}"][li][STD_IDX].cpu()), np.array(stats_t[im2_idx][f"stats_{st}"][li][STD_IDX].cpu())

                            # Calculate Jensen-Shannon Divergence
                            dist = compute_jsd(m1, s1, m2, s2)

                            # add to correct sum
                            if im2_idx in curr_style_indices:  # same style
                                similarity_dists += dist
                            else:
                                difference_dists += dist

                similarity_dists /= similiar_style_combs
                difference_dists /= different_style_combs
                
                if similarity_dists == 0 == difference_dists:
                    style_grade = 1
                else:
                    style_grade = (similarity_dists / difference_dists).item()
                    # grading[st][t].append(style_grade)
                grading[t][st].append(style_grade)

        mean_q.append(np.mean(grading[t]['q']))
        mean_k.append(np.mean(grading[t]['k']))
        mean_v.append(np.mean(grading[t]['v']))
        print(f"{t}: mean_q: {mean_q[-1]}; mean_k: {mean_k[-1]}; mean_v: {mean_v[-1]}")

        std_q.append(np.std(grading[t]['q']))
        std_k.append(np.std(grading[t]['k']))
        std_v.append(np.std(grading[t]['v']))
        print(f"{t}: std_q: {std_q[-1]}; std_k: {std_k[-1]}; std_v: {std_v[-1]}")

    # Save Gradings
    output_dir = os.path.join("outputs", 'statistics', 'layer_gradings')
    os.makedirs(output_dir, exist_ok=True)
    output_dict = {"layer_names": layer_names, "stats": grading}
    torch.save(output_dict, os.path.join(output_dir, series_name))

    # Plot Statistics
    # Mean Q
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_q, marker='o', linestyle='-', color='b', label='Mean Over Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Value')
    plt.title('Mean Query Values Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(input_dir, 'mean_q.png'))

    # Mean K
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_k, marker='o', linestyle='-', color='b', label='Mean Over Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Value')
    plt.title('Mean Key Values Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(input_dir, 'mean_k.png'))

    # Mean V
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_v, marker='o', linestyle='-', color='b', label='Mean Over Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Value')
    plt.title('Mean Key Values Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(input_dir, 'mean_v.png'))

    # Std Q
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, std_q, marker='o', linestyle='-', color='b', label='Std Over Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Value')
    plt.title('Std Query Values Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(input_dir, 'std_q.png'))

    # Std K
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, std_k, marker='o', linestyle='-', color='b', label='Std Over Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Value')
    plt.title('Std Key Values Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(input_dir, 'std_k.png'))

    # Std V
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, std_v, marker='o', linestyle='-', color='b', label='Std Over Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Value')
    plt.title('Std Key Values Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(input_dir, 'std_v.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_name", default=SERIES_NAME)
    args = parser.parse_args()
    main(args)


    
