import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator


def plot_metrics(tokens_seen, train_loss, val_loss, num_epochs):
    epochs = torch.linspace(0, num_epochs, len(train_loss))
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs, train_loss, label="training loss")
    ax1.plot(epochs, val_loss, linestyle="-.", label="validation loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("losses")
    ax1.legend(loc="upper right")

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_loss, alpha=0)
    ax2.set_xlabel("tokens seen")

    fig.tight_layout()
    plt.figure(dpi=500)
    plt.show()
