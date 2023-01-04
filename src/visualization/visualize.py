import logging
import random

import click
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.model import MyAwesomeModel
from src.models.dataset import MyDataset


@click.command()
@click.argument('model_checkpoint', type=click.Path(exists=True))
@click.argument('train_set', type=click.Path(exists=True))
@click.option('--samples', default=10000)
def main(model_checkpoint, train_set, samples):
    logger = logging.getLogger(__name__)

    logger.info('loading model')
    model = MyAwesomeModel()
    state = torch.load(model_checkpoint)
    model.load_state_dict(state)
    model.eval()

    logger.info('loading data')
    dataset = MyDataset(train_set)
    random.seed(0)
    indexes = random.sample(range(len(dataset)), samples)
    input_, target = dataset[indexes]

    logger.info('predicting')
    output = model.backbone(input_)
    output = output.reshape(output.size(0), -1)

    logger.info('reducing dimension')
    output = PCA(n_components=50).fit_transform(output.detach())
    output = TSNE().fit_transform(output)

    sns.set_theme()
    plt.figure()
    for i in range(10):
        x = output[target == i]
        sns.scatterplot(x=x[:, 0], y=x[:, 1])
    plt.savefig('reports/figures/mnist.png')
    plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
