import matplotlib.pyplot as plt


def plot_cvae_predictions(data_batch, train_dataset, trajectories,
                          transform_points,
                          n_samples_to_plot=10):

    n_samples_to_plot = min(n_samples_to_plot, trajectories.shape[2])
    samples_idx = iter(np.random.permutation(list(range(trajectories_batch.shape[0]))))

    fig, ax = plt.subplots(2, 3, figsize=(22, 10))

    # gt trajectories in pixels
    for j in range(2):
      for k in range(3):
        idx = next(samples_idx)
        img = data_batch['image'][idx, ...].numpy()
        img = img.transpose(1, 2, 0)
        img = train_dataset.rasterizer.to_rgb(img)
        target_pos_pix = transform_points(data_batch['target_positions'][idx].numpy(),
                                          data_batch['raster_from_agent'][idx].numpy())
        ax[j][k].imshow(img)
        # plot samples
        for i in range(n_samples_to_plot):
          sample = trajectories[idx, 0, i, :]
          sample = np.stack((sample[:50], sample[50:])).T
          sample_pos_pix = transform_points(sample, 
                                            data_batch["raster_from_agent"][idx].numpy())
          ax[j][k].plot(sample_pos_pix[:, 0], sample_pos_pix[:, 1], linewidth=2, alpha=0.5, c='white')

        # plot gt
        ax[j][k].scatter(target_pos_pix[:, 0], target_pos_pix[:, 1], s=15, c='r')
        ax[j][k].set_axis_off()
    fig.tight_layout()


def plot_extr_predictions(data_batch, train_dataset, trajectories,
                          transform_points,
                          n_samples_to_plot=10,
                          zoom=True):

    colors = ['tab:blue', 'tab:green', 'tab:olive']
    confs = [0.7, 0.2, 0.1] # TBD change to config
    sizes = [20, 15, 10]
    n_samples_to_plot = min(n_samples_to_plot, trajectories.shape[2])
    samples_idx = iter(np.random.permutation(list(range(trajectories_batch.shape[0]))))

    fig, ax = plt.subplots(2, 3, figsize=(22, 10))

    # gt trajectories in pixels
    for j in range(2):
      for k in range(3):
        idx = next(samples_idx)
        img = data_batch['image'][idx, ...].numpy()
        img = img.transpose(1, 2, 0)
        img = train_dataset.rasterizer.to_rgb(img)
        target_pos_pix = transform_points(data_batch['target_positions'][idx].numpy(),
                                          data_batch['raster_from_agent'][idx].numpy())
        ax[j][k].imshow(img)
        # plot samples
        for i in range(3):
          sample = trajectories[idx, i, :, :]
          sample_pos_pix = transform_points(sample.detach().cpu().numpy(), 
                                            data_batch["raster_from_agent"][idx].numpy())
          ax[j][k].scatter(sample_pos_pix[:, 0], sample_pos_pix[:, 1],
                           linewidth=2, alpha=1,
                           s=sizes[i],
                           c=colors[i],
                           label=confs[i])
        # plot gt
        ax[j][k].scatter(target_pos_pix[:, 0], target_pos_pix[:, 1], s=15, c='r', label='gt')

        ax[j][k].set_axis_off()
        ax[j][k].legend()
        if zoom:
          ax[j][k].set_xlim([30, 180])
          ax[j][k].set_ylim([30, 180])
    fig.tight_layout()