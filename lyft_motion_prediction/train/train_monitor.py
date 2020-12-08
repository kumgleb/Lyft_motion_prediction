import numpy as np
import matplotlib.pyplot as plt


def cvae_training_monitor(mse_losses, vlb_losses, criterion):

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    total_loss = np.array(mse_losses) + np.array(vlb_losses)
    ax[0].plot(np.arange(0, len(mse_losses)), total_loss)
    ax[1].plot(np.arange(0, len(vlb_losses)), mse_losses)
    ax[2].plot(np.arange(0, len(vlb_losses)), vlb_losses)

    ax[0].set_ylabel(f'MSE + {criterion} loss')
    ax[0].set_xlabel('Iteration')
    ax[0].set_yscale('log')
    ax[0].grid('on')

    ax[1].set_ylabel('MSE')
    ax[1].set_xlabel('Iteration')
    ax[1].set_yscale('log')
    ax[1].grid('on')

    ax[2].set_ylabel(f'{criterion} loss')
    ax[2].set_xlabel('Iteration')
    ax[2].set_yscale('log')
    ax[2].grid('on')

    fig.tight_layout()
    plt.show()


def extractor_training_monitor(losses_train):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(np.arange(0, len(losses_train)), losses_train)
    ax.set_yscale('log')
    ax.grid()
    plt.show()
