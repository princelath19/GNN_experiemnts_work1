import subprocess
import os

#first-one was with heat equation  - saved in results_best_approx - run_output.txt
# =====================================================================
# 1. Define all dataset-specific hyperparameters
# =====================================================================

CONFIGS = {
    "Cora": {
       # "--num_layers": 2,
        "--input_dropout": "0.",
        "--decoder_layers": 1,
        "--encoder_layers": 2,
        "--decoder_dropout": "0.",
        "--hidden_channels": 64,
        "--sparsity": "0.",
        "--self_loops": "1.",
        "--undirected": "",
        "--patience": 200,
        "--learning_rate": "0.01",
        "--weight_decay": "0.005",
        "--normalize_features": "",
        "--norm_ord": "inf",
        "--norm_dim": "1",
        "--lcc": ""
    },

    "film": {
        # film has two variants: directed and Tolokers, but you gave directed here
        #"--num_layers": 1,
        "--encoder_layers": 3,
        "--decoder_layers": 2,
        "--hidden_channels": 256,
        "--sparsity": "0.",
        "--input_dropout": "0.",
        "--decoder_dropout": "0.1",
        "--norm_ord": "2",
        "--norm_dim": "1",
        "--sklearn": "",
        "--learning_rate": "0.001",
        "--weight_decay": "0.0005",
        "--lcc": ""
    },

    "Tolokers": {
        #"--num_layers": 3,
        "--decoder_layers": 2,
        "--encoder_layers": 2,
        "--hidden_channels": 512,
        "--sparsity": "0.",
        "--learning_rate": "0.001",
        "--weight_decay": "0.",
    },
    "chameleon": {
       # "--num_layers": 5,
        "--input_dropout": "0.",
        "--decoder_layers": 2,
        "--encoder_layers": 1,
        "--decoder_dropout": "0.",
        "--hidden_channels": 64,
        "--sparsity": "0.",
        "--self_loops": "0.",
        "--patience": 100,
        "--learning_rate": "0.01",
        "--weight_decay": "0.001",
        "--normalize_features": "",
        "--norm_ord": "2",
        "--norm_dim": "0",
        "--lcc": ""
    }
}

dataset = "Cora"
Ns = [5, 10, 15, 30]
dts = [round(0.1 * i, 1) for i in range(1, 11)]   # 0.1 to 1.0
#Ns = [30]
#dts = [0.7,0.8,0.9,1.0]
# =====================================================================
# 3. Organized folder structure
# =====================================================================

root_dir = "Experiment_1"
method_dir = os.path.join(root_dir, "FE")
dataset_dir = os.path.join(method_dir, "Cora_non_activated_L_exp_1")

os.makedirs(dataset_dir, exist_ok=True)

# =====================================================================
# 4. Run experiments
# =====================================================================

config = CONFIGS[dataset]

for N in Ns:
    N_dir = os.path.join(dataset_dir, f"N_{N}")
    os.makedirs(N_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Running N = {N}")
    print("=" * 80)

    for dt in dts:
        dt_dir = os.path.join(N_dir, f"dt_{dt:.1f}")
        os.makedirs(dt_dir, exist_ok=True)

        logfile = os.path.join(dt_dir, "run_output.txt")

        cmd = [
            "python", "node_classification.py",
            "--dataset", dataset,
            "--equation", "h",
            "--num_layers", str(N),
            "--step_size", str(dt),
            "--exponent", "1.0",
            "--spectral_shift", "0.0"
        ]

        for key, value in config.items():
            cmd.append(key)
            if value != "":
                cmd.append(str(value))

        print(f"Running: dataset={dataset}, N={N}, dt={dt:.1f}, T={N*dt:.1f}")

        with open(logfile, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

        print(f"Saved log to: {logfile}")

print("\nAll experiments finished.\n")
