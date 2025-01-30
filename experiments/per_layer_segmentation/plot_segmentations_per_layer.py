import pandas as pd 
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load up the pandas dataframe
    data_path = "results/per_layer_metrics.csv"
    df = pd.read_csv(data_path)
    # df = pd.DataFrame(columns=["image_id", "layer", "pixelwise_average", "mIoU", "AP"])
    # Plot the results
    fig, axs = plt.subplots(1, 1, figsize=(6 * 0.9, 4 * 0.9))
    # Average each metric over the layers
    df_stds = df.groupby("layer").std().reset_index()
    df = df.groupby("layer").mean().reset_index()
    # Also use error bounds
    # Plot the layer vs pixelwise average
    axs.plot(df["layer"], df["pixelwise_average"], label="Accuracy")
    axs.fill_between(
        df["layer"],
        df["pixelwise_average"] - df_stds["pixelwise_average"] ** 2,
        df["pixelwise_average"] + df_stds["pixelwise_average"] ** 2,
        alpha=0.2
    )
    axs.set_title("Layer vs Segmentation Performance")
    axs.set_xlabel("Layer")
    axs.set_ylabel("Metric")
    axs.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # axs[0].grid()
    # Plot the layer vs mIoU
    axs.plot(df["layer"], df["mIoU"], label="mIoU")
    axs.fill_between(
        df["layer"],
        df["mIoU"] - df_stds["mIoU"] ** 2,
        df["mIoU"] + df_stds["mIoU"] ** 2,
        alpha=0.2
    )   
    # axs.set_title("Layer vs mean Intersection over Union")
    axs.set_xlabel("Layer")
    # axs.set_ylabel("mean Intersection over Union")
    axs.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # axs[1].grid()
    # Plot the layer vs AP
    axs.plot(df["layer"], df["AP"], label="mAP")
    axs.fill_between(
        df["layer"],
        df["AP"] - df_stds["AP"] ** 2,
        df["AP"] + df_stds["AP"] ** 2,
        alpha=0.2
    )
    # axs.set_title("Layer vs Average Precision")
    axs.set_xlabel("Layer")
    # axs.set_ylabel("Average Precision")
    axs.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Now plot lines for each of the three metrics
    combined_mAP = 0.9055
    combined_mIoU = 0.6708
    combined_accuracy = 0.8300
    # Use same color as each metric for the line
    axs.axhline(y=combined_accuracy, color="C0", linestyle="--", label="Combined Accuracy")
    axs.axhline(y=combined_mIoU, color="C1", linestyle="--", label="Combined mIoU")
    axs.axhline(y=combined_mAP, color="C2", linestyle="--", label="Combined mAP")

    axs.set_xlim(0, 18)
    axs.set_ylim(0.24, 0.93)
    # axs[2].grid()
    axs.legend(ncol=2)

    plt.tight_layout()

    plt.savefig("results/per_layer_metrics.png", dpi=300)
    plt.savefig("results/per_layer_metrics.svg")
    plt.savefig("results/per_layer_metrics.pdf")