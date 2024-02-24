import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(arr, path):
    print(f"Min: {np.min(arr)}, Max: {np.max(arr)}")
    _ = plt.hist(arr, bins='auto')
    plt.savefig(path)
    plt.close()

def plot_bar_plot(err_arr, seq_lens, path):
    seq_means = []
    full_mean = np.mean(err_arr) # Use this
    print(f"Min: {np.min(err_arr)}, Max: {np.max(err_arr)}")

    seq_wise_data = []
    start_idx = 0
    end_idx = seq_lens[0]
    for i in range(0, len(seq_lens)-1):
        end_idx = start_idx+seq_lens[i]
        seq_wise_data.append(err_arr[start_idx:end_idx])
        seq_means.append(np.mean(err_arr[start_idx:end_idx]))
        print(np.mean(err_arr[start_idx:end_idx] == seq_wise_data[i]))
        start_idx = end_idx

    end_idx = start_idx+seq_lens[-1]
    seq_means.append(np.mean(err_arr[start_idx:end_idx]))
    
    fig = plt.figure(figsize = (10, 5))
    # creating the bar plot
    seq_idx = np.arange(len(seq_lens))
    np.save(path+".npy", np.array(seq_means))
    joblib.dump(path+".pkl", seq_wise_data)
    plt.bar(seq_idx, seq_means)
    plt.savefig(path+".png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--target_dataset', help='Description for foo argument', required=True)
    parser.add_argument('-m','--target_metric', help='Description for foo argument', required=True)
    args = parser.parse_args()
    target_dataset =  args.target_dataset
    target_metric =  args.target_metric
    # target_dataset = "3dpw"
    print("Dataset: ", target_dataset)
    SAVE_SEQ_LENS = False
    # pkl_path = f"../output/error_arrays/glot_result_{target_dataset}.pkl"
    npy_path = f"output/error_arrays/{target_metric}_{target_dataset}.npy"

    # d = joblib.load(pkl_path)

    # mpjpe_data = d['mpjpe']
    # pampjpe_data = d['mpjpe_pa']
    # acc_err_data = d['accel_err']
    # if target_dataset == "3dpw":
    #     mpvpe_data = d['mpvpe']

    # seq_lens = []
    # if SAVE_SEQ_LENS == True:
    #     mpjpe_vals_all = []
    #     for i in range(len(mpjpe_data)):
    #         seq_lens.append(mpjpe_data[i].shape[0])
    #     seq_lens = np.array(seq_lens)
    #     np.save(f"{target_dataset}_seq_lens.npy", seq_lens)
    #     print(seq_lens.shape)
    # else:
    #     seq_lens = np.load(f"{target_dataset}_seq_lens.npy")
    seq_lens = np.load(f"../GLoT/{target_dataset}_seq_lens.npy")

    npy_path = f"output/error_arrays/{target_metric}_{target_dataset}.npy"
    err_vals = np.load(npy_path, allow_pickle=True)
    err_vals = np.mean(err_vals, 1)
    print("Num frames: ", err_vals.shape)

    # mpjpe_vals = np.concatenate(mpjpe_data)
    # mpjpe_vals = np.load()
    # pampjpe_vals = np.concatenate(pampjpe_data)
    # acc_err_vals = np.concatenate(acc_err_data)
    # if target_dataset == "3dpw":
    #     mpvpe_vals = np.concatenate(mpvpe_data)

    # plot_hist(mpjpe_vals, f"output/error_arrays/{target_dataset}_mpjpe_hist_full.png")
    # plot_hist(pampjpe_vals, f"output/error_arrays/{target_dataset}_pampjpe_hist_full.png")
    # plot_hist(acc_err_vals, f"output/error_arrays/{target_dataset}_acc_err_hist_full.png")
    # if target_dataset == "3dpw":
    #     plot_hist(mpvpe_vals, f"output/error_arrays/{target_dataset}_mpvpe_full.png")
    plot_hist(err_vals, f"output/error_arrays/{target_dataset}_{target_metric}_hist.png")
    
    # plot_bar_plot(mpjpe_data, f"../output/error_arrays/{target_dataset}_mpjpe_bar_plot.png")
    # plot_bar_plot(pampjpe_data, f"../output/error_arrays/{target_dataset}_pampjpe_bar_plot.png")
    # plot_bar_plot(acc_err_data, f"../output/error_arrays/{target_dataset}_acc_err_bar_plot.png")
    # if target_dataset == "3dpw":
    #     plot_all_errors(mpvpe_data, f"../output/error_arrays/{target_dataset}_mpvpe_bar_plot.png")
    plot_bar_plot(err_vals, seq_lens, f"output/error_arrays/{target_dataset}_{target_metric}_bar_plot")   