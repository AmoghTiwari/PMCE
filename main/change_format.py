import numpy as np
import joblib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--target_dataset', help='Description for foo argument', required=True)
    args = parser.parse_args()
    target_dataset =  args.target_dataset

    seq_lens = np.load(f"../GLoT/{target_dataset}_seq_lens.npy")
    mpjpe_path = f"output/error_arrays/mpjpe_{target_dataset}.npy"
    mpjpe_vals = np.load(mpjpe_path)
    pampjpe_path = f"output/error_arrays/pampjpe_{target_dataset}.npy"
    pampjpe_vals = np.load(pampjpe_path)
    acc_err_path = f"output/error_arrays/acc_error_{target_dataset}_v3.npy"
    acc_err_vals = np.load(acc_err_path)
    if target_dataset == "3dpw":
        mpvpe_path = f"output/error_arrays/mpjpe_{target_dataset}.npy"
        mpvpe_vals = np.load(mpvpe_path)
    
    seq_lens = np.load(f"../GLoT/{target_dataset}_seq_lens.npy")

    mpjpe_seq_wise = []
    pampjpe_seq_wise = []
    acc_err_seq_wise = []
    mpvpe_seq_wise = []
    start_idx = 0
    end_idx = seq_lens[0]

    for i in range(0, len(seq_lens)):
        end_idx = start_idx+seq_lens[i]
        mpjpe_seq_wise.append(np.mean(mpjpe_vals[start_idx:end_idx], 1))
        pampjpe_seq_wise.append(np.mean(pampjpe_vals[start_idx:end_idx], 1))
        acc_err_seq_wise.append((acc_err_vals[start_idx:end_idx]))
        if target_dataset == "3dpw":
            mpvpe_seq_wise.append(np.mean(mpvpe_vals[start_idx:end_idx], 1))
        start_idx = end_idx
    all_data = {}
    all_data['mpjpe'] = mpjpe_seq_wise
    all_data['mpjpe_pa'] = pampjpe_seq_wise
    all_data['accel_err'] = acc_err_seq_wise
    if target_dataset == "3dpw":
        all_data['mpvpe'] = mpvpe_seq_wise
    joblib.dump(all_data, f"output/error_arrays/pmce_result_{target_dataset}.pkl")