import pandas as pd 
import matplotlib.pyplot as plt

def plot_all_in_one(
    df,
    save_path: str = "results/per_time_metrics.png"
):
    # Compute the mean and std of each metric over the timesteps
    df_std = df.groupby("timestep").std().reset_index()
    df = df.groupby("timestep").mean().reset_index()
    # Multiple timestep by 20
    df["timestep"] = df["timestep"] * 20
    # Plot the results
    fig, axs = plt.subplots(1, 1, figsize=(6*0.9, 4*0.9))
    # Compute the error bounds
    # Plot the layer vs pixelwise average
    axs.plot(df["timestep"], df["pixelwise_average"], label="Pixelwise Accuracy")
    axs.fill_between(
        df["timestep"],
        df["pixelwise_average"] - df_std["pixelwise_average"] ** 2,
        df["pixelwise_average"] + df_std["pixelwise_average"] ** 2,
        alpha=0.2,
    )
    # Plot the layer vs mIoU
    axs.plot(df["timestep"], df["mIoU"], label="mIoU")
    axs.fill_between(
        df["timestep"],
        df["mIoU"] - df_std["mIoU"] ** 2,
        df["mIoU"] + df_std["mIoU"] ** 2,
        alpha=0.2,
    )
    # Plot the layer vs AP
    axs.plot(df["timestep"], df["AP"], label="Average Precision")
    axs.fill_between(
        df["timestep"],
        df["AP"] - df_std["AP"] ** 2,
        df["AP"] + df_std["AP"] ** 2,
        alpha=0.2,
    )
    axs.set_title("Diffusion Timestep vs Segmentation Performance")
    axs.set_xlabel("Timestep")
    axs.set_ylabel("Metric Value")
    axs.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # axs[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))


     # Now plot lines for each of the three metrics
    combined_mAP = 0.9055
    combined_mIoU = 0.6708
    combined_accuracy = 0.8300
    # Use same color as each metric for the line
    axs.axhline(y=combined_accuracy, color="C0", linestyle="--", label="Combined Accuracy")
    axs.axhline(y=combined_mIoU, color="C1", linestyle="--", label="Combined mIoU")
    axs.axhline(y=combined_mAP, color="C2", linestyle="--", label="Combined mAP")

    plt.text(0, 0.148, 'All Noise', fontsize=10, color='black', ha='center', va='bottom')
    plt.text(1000, 0.148, 'No Noise', fontsize=10, color='black', ha='center', va='bottom')

    axs.set_ylim(0.24, 0.93)
    axs.set_xlim(0, 980)

    # Add the legend
    axs.legend(framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("results/per_time_metrics.svg")
    plt.savefig("results/per_time_metrics.pdf")


if __name__ == "__main__":
    # Load up the pandas dataframe
    data_path = "results/per_time_metrics.csv"
    df = pd.read_csv(data_path)
    plot_all_in_one(df)
    # df = pd.DataFrame(columns=["image_id", "layer", "pixelwise_average", "mIoU", "AP"])
    # Plot the results
    # fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    # # Get the standard deviations of each metric for each time step
    # df_std = df.groupby("timestep").std().reset_index()
    # # Average each metric over the layers
    # df = df.groupby("timestep").mean().reset_index()
    # print(df_std)
    # # # Also use error bounds
    # # Plot the layer vs pixelwise average
    # axs[0].plot(df["timestep"], df["pixelwise_average"])
    # axs[0].fill_between(
    #     df["timestep"],
    #     df["pixelwise_average"] - df_std["pixelwise_average"] ** 2,
    #     df["pixelwise_average"] + df_std["pixelwise_average"] ** 2,
    #     alpha=0.2
    # )
    # axs[0].set_title("Timestep vs Pixelwise Accuracy")
    # axs[0].set_xlabel("Timestep")
    # axs[0].set_ylabel("Pixelwise Average")
    # axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # axs[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # # axs[0].grid()
    # # Plot the layer vs mIoU
    # axs[1].plot(df["timestep"], df["mIoU"])
    # axs[1].fill_between(
    #     df["timestep"],
    #     df["mIoU"] - df_std["mIoU"] ** 2,
    #     df["mIoU"] + df_std["mIoU"] ** 2,
    #     alpha=0.2
    # )
    # axs[1].set_title("Timestep vs mean Intersection over Union")
    # axs[1].set_xlabel("Timestep")
    # axs[1].set_ylabel("mean Intersection over Union")
    # axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # axs[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # # axs[1].grid()
    # # Plot the layer vs AP
    # axs[2].plot(df["timestep"], df["AP"])
    # axs[2].fill_between(
    #     df["timestep"],
    #     df["AP"] - df_std["AP"] ** 2,
    #     df["AP"] + df_std["AP"] ** 2,
    #     alpha=0.2
    # )
    # axs[2].set_title("Timestep vs Average Precision")
    # axs[2].set_xlabel("Timestep")
    # axs[2].set_ylabel("Average Precision")
    # axs[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # axs[2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # # axs[2].grid()

    # plt.tight_layout()

    # plt.savefig("results/per_time_metrics.png", dpi=300)