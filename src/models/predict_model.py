import logging

import click
import torch

from model import MyAwesomeModel
from dataset import MyDataset


@click.command()
@click.argument('model_checkpoint', type=click.Path(exists=True))
@click.argument('test_set', type=click.Path(exists=True))
def main(model_checkpoint, test_set):
    logger = logging.getLogger(__name__)

    logger.info('loading model')
    model = MyAwesomeModel()
    state = torch.load(model_checkpoint)
    model.load_state_dict(state)
    model.eval()

    logger.info('loading data')
    dataset = MyDataset(test_set)
    input_, target = dataset[:]

    logger.info('predicting')
    output = model(input_).argmax(dim=-1)
    acc = torch.sum(output == target)/len(output)

    logger.info(f'accuracy: {acc:.2f}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
