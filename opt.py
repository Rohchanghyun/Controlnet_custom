import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="../../../dataset/emoji_data",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis', 'tsne'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='/workspace/dataset/top15character/Elric Edward/Image_2.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--extractor_weight',
                    default=None,
                    help='load weights ')

parser.add_argument('--image_adapter_weight',
                    default=None,
                    help='load weights ')

parser.add_argument('--epoch',
                    default=3000,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=8,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=8,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    help='the batch size for test')

parser.add_argument("--optimizer",
                    default='adamw',
                    help='optimizer')

parser.add_argument("--output_dir",
                    default='../../../output/256/',
                    help='output_dir')

opt = parser.parse_args()
