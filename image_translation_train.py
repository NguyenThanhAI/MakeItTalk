import os
import argparse

from image_translation import ImageTranslationConfig, ImageTranslation


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate_decay_type", type=str, default="constant")
    parser.add_argument("--decay_steps", type=int, default=20000)
    parser.add_argument("--decay_rate", type=float, default=0.85)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lambda_a", type=float, default=1.)
    parser.add_argument("--use_cycle_loss", type=str2bool, default=False)
    parser.add_argument("--use_discriminator", type=str2bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--is_loadmodel", type=str2bool, default=False)
    parser.add_argument("--per_process_gpu_memory_fraction", type=float, default=1.0)
    parser.add_argument("--summary_dir", type=str, default="summary")
    parser.add_argument("--model_dir", type=str, default="saved_model")
    parser.add_argument("--vgg_model_dir", type=str, default=r"D:\vgg_19_2016_08_28")
    parser.add_argument("--vgg_checkpoint", type=str, default="vgg_19.ckpt")
    parser.add_argument("--generator_checkpoint_name", type=str, default=None)
    parser.add_argument("--discriminator_checkpoint_name", type=str, default=None)
    parser.add_argument("--discriminator_update_steps", type=int, default=4)
    parser.add_argument("--dis_gen_learning_rate_ratio", type=float, default=2.)
    parser.add_argument("--dataset_path", type=str, default=r"D:\Face_Animation\tfrecord\training.tfrecord")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=250)
    parser.add_argument("--summary_frequency", type=int, default=25)
    parser.add_argument("--save_network_frequency", type=int, default=2500)
    parser.add_argument("--is_training", type=str2bool, default=True)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--gpu_devices", type=str, default="1")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    print("Arguments: {}".format(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    config = ImageTranslationConfig(input_size=args.input_size,
                                    learning_rate=args.learning_rate,
                                    learning_rate_decay_type=args.learning_rate_decay_type,
                                    decay_steps=args.decay_steps,
                                    decay_rate=args.decay_rate,
                                    momentum=args.momentum,
                                    lambda_a=args.lambda_a,
                                    use_cycle_loss=args.use_cycle_loss,
                                    use_discriminator=args.use_discriminator,
                                    weight_decay=args.weight_decay,
                                    is_loadmodel=args.is_loadmodel,
                                    per_process_gpu_memory_fraction=args.per_process_gpu_memory_fraction,
                                    summary_dir=args.summary_dir,
                                    model_dir=args.model_dir,
                                    vgg_model_dir=args.vgg_model_dir,
                                    vgg_checkpoint=args.vgg_checkpoint,
                                    generator_checkpoint_name=args.generator_checkpoint_name,
                                    discriminator_checkpoint_name=args.discriminator_checkpoint_name,
                                    discriminator_update_steps=args.discriminator_update_steps,
                                    dis_gen_learning_rate_ratio=args.dis_gen_learning_rate_ratio,
                                    dataset_path=args.dataset_path,
                                    batch_size=args.batch_size,
                                    num_epochs=args.num_epochs,
                                    summary_frequency=args.summary_frequency,
                                    save_network_frequency=args.save_network_frequency,
                                    is_training=args.is_training,
                                    optimizer=args.optimizer)

    image_translation = ImageTranslation(config=config)

    image_translation.train()
