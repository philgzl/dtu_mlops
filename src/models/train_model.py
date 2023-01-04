import os
import logging

import click
import torch
import matplotlib.pyplot as plt

from model import MyAwesomeModel
from dataset import MyDataset


@click.command(context_settings={'show_default': True})
@click.argument('train_set', type=click.Path(exists=True))
@click.option("--lr", default=1e-3)
@click.option("--bs", default=128)
@click.option("--shuffle", is_flag=True)
@click.option("--workers", default=0)
@click.option("--epochs", default=10)
def main(train_set, lr, bs, shuffle, workers, epochs):
    logger = logging.getLogger(__name__)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info('initializing dataset')
    dataset = MyDataset(train_set)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
    )

    logger.info('initializing model')
    model = MyAwesomeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info('starting training loop')
    loss_curve = []
    for epoch in range(epochs):
        total_loss = 0
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(dataloader)
        loss_curve.append(avg_loss)
        logger.info(f'Epoch {epoch}; loss: {avg_loss:.4f}')

    outdir = 'models'
    outfile = 'trained_model.pt'
    torch.save(model.state_dict(), os.path.join(outdir, outfile))

    plt.plot(loss_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plotfile = "training_curve.png"
    plt.savefig(os.path.join(outdir, plotfile))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
