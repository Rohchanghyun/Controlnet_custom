import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="/workspace/data/changhyun/dataset/emo_sketch/",
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
                    default="/workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt",
                    help='load weights ')

parser.add_argument('--image_adapter_weight',
                    default="None",
                    help='load weights ')

parser.add_argument('--epoch',
                    type=int,
                    default=2000,
                    help='number of epoch to train')

parser.add_argument('--emo_sketch_path', type=str, default='./emo_sketch',
                    help='emo_sketch 데이터셋 루트 경로')

parser.add_argument('--lr',
                    type=float,
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    type=list,
                    default=[320, 380],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument('--clip_embeddings_dim',
                    type=int,
                    default=768,
                    help='clip embeddings dim')

parser.add_argument('--model_id',
                    type=str,
                    default="neta-art/neta-xl-2.0",
                    help='모델 ID (SD 1.5 또는 SDXL)')

parser.add_argument('--vae_model_path',
                    type=str,
                    default=None,
                    help='VAE 모델 경로')

parser.add_argument("--batchid",
                    type=int,
                    default=8,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    type=int,
                    default=8,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    type=int,
                    default=8,
                    help='the batch size for test')

parser.add_argument("--optimizer",
                    default='adamw',
                    help='optimizer')

parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-4,
                    help='weight decay for optimizer')

parser.add_argument("--output_dir",
                    default='../../../output/256/',
                    help='output_dir')

parser.add_argument('--validation_image_path',
                    type=str,
                    default="/path/to/your/validation/image.png",
                    help='path to the validation image')

parser.add_argument('--validation_caption',
                    type=str,
                    default="your caption",
                    help='caption for the validation image')

parser.add_argument('--validation_num_samples',
                    type=int,
                    default=1,
                    help='number of samples for validation')

parser.add_argument('--guidance_scale_1',
                    type=float,
                    default=1.5,
                    help='guidance scale for validation')

parser.add_argument('--guidance_scale_2',
                    type=float,
                    default=7.5,
                    help='guidance scale for validation')

parser.add_argument('--inference_height',
                    type=int,
                    default=1024,
                    help='inference height for validation')

parser.add_argument('--inference_width',
                    type=int,
                    default=1024,
                    help='inference width for validation')

parser.add_argument('--num_inference_steps',
                    type=int,
                    default=50,
                    help='number of inference steps for validation')

opt = parser.parse_args()
