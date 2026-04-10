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

# =====================================================================
# 2. Select datasets to run
# =====================================================================

datasets = ["Cora"]
number_Layer = ["4"]

# =====================================================================
# 3. Create master results directory
# =====================================================================

#results_root = "results_best_approx_activation_Tanh_updated"
results_root = "Experiment"
#results_root = "results_best_approx_activation_LeakyReLU_updated"
#results_root = "results_best_approx"
os.makedirs(results_root, exist_ok=True)

# =====================================================================
# 4. Run all experiments
# =====================================================================

for dataset in datasets:
#for cora, we do not need number of layers = 2, as it is already done!!.
#for toloker, we do not need number of layers =2, as it is already done!!.
    print("\n" + "="*80)
    print(f"Running dataset: {dataset}")
    print("="*80)



    # Create folder for this dataset
    dataset_dir = os.path.join(results_root, f"{dataset}_best_approx_high_layers_no_activation")
    #dataset_dir = os.path.join(results_root, f"{dataset}_best_approx")
    os.makedirs(dataset_dir, exist_ok=True)

    for layer in number_Layer:
        #print("\n" + "=" * 80)
        print(f"Running number of layers: {layer}")
        #print("=" * 80)

    # Save file inside the dataset folder
    #outfile = os.path.join(dataset_dir, "run_output_num_layers_3.txt")
        outfile = os.path.join(dataset_dir, f"run_output_number_layers_{layer}.txt")

        # Start building the command
        #cmd = ["python", "node_classification.py", "--dataset", dataset, "--equation", "h", "--num_layers", "3"]
        cmd = ["python", "node_classification.py", "--dataset", dataset, "--equation", "h", "--num_layers", layer, "--step_size", "1.0","--exponent","1.0"]
        #cmd = ["python", "node_classification.py", "--dataset", dataset]

        # Add dataset-specific args
        config = CONFIGS[dataset]

        for key, value in config.items():
            cmd.append(key)
            if value != "":   # flags without value
                cmd.append(str(value))

        # Run and save logs
        with open(outfile, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

            print(f"  ✅ Results saved to {outfile}")

print("\n🎯 ALL EXPERIMENTS FINISHED SUCCESSFULLY! \n")


