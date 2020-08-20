from image_translation import ImageTranslationConfig, ImageTranslation

config = ImageTranslationConfig(dataset_path=r"D:\Face_Animation\tfrecord\training.tfrecord",
                                vgg_model_dir=r"D:\vgg_19_2016_08_28",
                                vgg_checkpoint="vgg_19.ckpt")

image_translation = ImageTranslation(config=config)

image_translation.train()
