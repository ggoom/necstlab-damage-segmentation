import shutil
import os
import random
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from image_utils import TensorBoardImage, ImagesAndMasksGenerator, trainGenerator
import git
from gcp_utils import copy_folder_locally_if_missing

from models import generate_compiled_segmentation_model, generate_compiled_3d_segmentation_model
from unet3d.generator import get_training_and_validation_generators
from unet3d.data import write_data_to_file, open_data_file

metadata_file_name = 'metadata.yaml'
tmp_directory = Path('./tmp')


def sample_image_and_mask_paths(generator, n_paths):
    random.seed(0)
    rand_inds = [random.randint(0, len(generator.image_filenames)-1) for _ in range(n_paths)]
    image_paths = list(np.asarray(generator.image_filenames)[rand_inds])
    mask_paths = list(np.asarray(generator.mask_filenames)[rand_inds])
    # mask_paths = [{c: list(np.asarray(generator.mask_filenames[c]))[i] for c in generator.mask_filenames} for i in rand_inds]
    return list(zip(image_paths, mask_paths))


def train(gcp_bucket, config_file):

    start_dt = datetime.now()

    with Path(config_file).open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    assert "gs://" in gcp_bucket

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    local_dataset_dir = Path('dendrites3d', 'datasets')
    # GOOFYS mounted bucket replaced this
    # local_dataset_dir = Path(tmp_directory, 'datasets')
    # copy_folder_locally_if_missing(os.path.join(gcp_bucket, 'datasets', train_config['dataset_id']),
    #                                local_dataset_dir)

    model_id = "{}_{}".format(train_config['model_id_prefix'], datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'))
    model_dir = Path(tmp_directory, 'models', model_id)
    model_dir.mkdir(parents=True)

    plots_dir = Path(model_dir, 'plots')
    plots_dir.mkdir(parents=True)

    logs_dir = Path(model_dir, 'logs')
    logs_dir.mkdir(parents=True)

    data_dir = Path(tmp_directory, 'data_files')
    data_dir.mkdir(parents=True)

    with Path(local_dataset_dir, train_config['dataset_id'], 'config.yaml').open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    with Path(model_dir, 'config.yaml').open('w') as f:
        yaml.safe_dump({'train_config': train_config}, f)

    target_size = dataset_config['target_size']
    batch_size = train_config['batch_size']
    epochs = 1
    generator_type = train_config['generator_type']
    augmentation_type = train_config['data_augmentation']['augmentation_type']

    if generator_type == '3D':
        config = dict()

        config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
        config["image_shape"] = (20, 512, 512)  # This determines what shape the images will be cropped/resampled to.
        config["patch_shape"] = None  # switch to None to train on the whole image
        config["labels"] = (1,)  # the label numbers on the input image
        config["n_labels"] = len(config["labels"])
        config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
        config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
        config["nb_channels"] = len(config["training_modalities"])
        if "patch_shape" in config and config["patch_shape"] is not None:
            config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
        else:
            config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
        config["truth_channel"] = config["nb_channels"]
        config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

        config["batch_size"] = 1
        config["validation_batch_size"] = 1
        config["n_epochs"] = 1  # cutoff the training after this many epochs
        config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
        config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
        config["initial_learning_rate"] = 0.00001
        config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
        config["validation_split"] = 0.8  # portion of the data that will be used for training
        config["flip"] = False  # augments the data by randomly flipping an axis during
        config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
        config["distort"] = None  # switch to None if you want no distortion
        config["augment"] = config["flip"] or config["distort"]
        config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
        config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
        config["skip_blank"] = True  # if True, then patches without any target will be skipped

        config["training_data_file"] = Path(data_dir, 'training.h5').as_posix()
        config["validation_data_file"] = Path(data_dir, 'validation.h5').as_posix()
        # config["model_file"] = os.path.abspath("tumor_segmentation_model.h5")
        config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.

        training_dataset_directory = Path(local_dataset_dir, train_config['dataset_id'], 'train').as_posix()
        training_image_filenames = sorted(Path(training_dataset_directory, 'images').iterdir())
        training_mask_filenames = sorted(Path(training_dataset_directory, 'masks').iterdir())

        training_data_files = [(image, mask) for image, mask in zip(training_image_filenames, training_mask_filenames)]
        write_data_to_file(training_data_files, config["training_data_file"], image_shape=config["image_shape"], crop=False)
        training_data_file_opened = open_data_file(config["training_data_file"])

        validation_dataset_directory = Path(local_dataset_dir, train_config['dataset_id'], 'validation').as_posix()
        validation_image_filenames = sorted(Path(validation_dataset_directory, 'images').iterdir())
        validation_mask_filenames = sorted(Path(validation_dataset_directory, 'masks').iterdir())
        validation_data_files = [(image, mask) for image, mask in zip(validation_image_filenames, validation_mask_filenames)]
        write_data_to_file(validation_data_files, config["validation_data_file"], image_shape=config["image_shape"], crop=False)
        validation_data_file_opened = open_data_file(config["validation_data_file"])

        train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            training_data_file_opened,
            validation_data_file_opened,
            batch_size=config["batch_size"],
            data_split=config["validation_split"],
            overwrite=False,
            n_labels=config["n_labels"],
            labels=config["labels"],
            patch_shape=config["patch_shape"],
            validation_batch_size=config["validation_batch_size"],
            validation_patch_overlap=config["validation_patch_overlap"],
            training_patch_start_offset=config["training_patch_start_offset"],
            permute=config["permute"],
            augment=config["augment"],
            skip_blank=config["skip_blank"],
            augment_flip=config["flip"],
            augment_distortion_factor=config["distort"])
    elif augmentation_type == 'necstlab':  # necstlab's workflow
        train_generator = ImagesAndMasksGenerator(
            Path(local_dataset_dir, train_config['dataset_id'], 'train').as_posix(),
            rescale=1./255,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=True,
            random_rotation=train_config['data_augmentation']['necstlab_augmentation']['random_90-degree_rotations'],
            seed=train_config['training_data_shuffle_seed'])

        validation_generator = ImagesAndMasksGenerator(
            Path(local_dataset_dir, train_config['dataset_id'],
                 'validation').as_posix(),
            rescale=1./255,
            target_size=target_size,
            batch_size=batch_size)
    elif augmentation_type == 'bio':  # new workflow
        bio_augmentation = train_config['data_augmentation']['bio_augmentation']
        augmentation_dict = dict(rotation_range=bio_augmentation['rotation_range'],
                                 width_shift_range=bio_augmentation['width_shift_range'],
                                 height_shift_range=bio_augmentation['height_shift_range'],
                                 shear_range=bio_augmentation['shear_range'],
                                 zoom_range=bio_augmentation['zoom_range'],
                                 horizontal_flip=bio_augmentation['horizontal_flip'],
                                 fill_mode=bio_augmentation['fill_mode'],
                                 cval=0)
        train_generator = trainGenerator(
            batch_size=batch_size,
            train_path=Path(local_dataset_dir, train_config['dataset_id'], 'train').as_posix(),
            image_folder='images',
            mask_folder='masks',
            aug_dict=augmentation_dict,
            target_size=target_size,
            seed=train_config['training_data_shuffle_seed'])

        validation_generator = trainGenerator(
            batch_size=batch_size,
            train_path=Path(local_dataset_dir, train_config['dataset_id'], 'validation').as_posix(),
            image_folder='images',
            mask_folder='masks',
            aug_dict=augmentation_dict,
            target_size=target_size,
            seed=train_config['training_data_shuffle_seed'])

    if generator_type == '2D':
        compiled_model = generate_compiled_segmentation_model(
            train_config['segmentation_model']['model_name'],
            train_config['segmentation_model']['model_parameters'],
            1,
            train_config['loss'],
            train_config['optimizer'])
    elif generator_type == '3D':
        compiled_model = generate_compiled_3d_segmentation_model(
            (1, 20, 512, 512),  # config["image_shape"],
            n_labels=1,
            n_base_filters=4,
            depth=2,
        )

        print(compiled_model.summary())

    model_checkpoint_callback = ModelCheckpoint(Path(model_dir, 'model.hdf5').as_posix(),
                                                monitor='loss', verbose=1, save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=logs_dir.as_posix(), write_graph=True,
                                       write_grads=False, write_images=True, update_freq='epoch', profile_batch=0)

    # n_sample_images = 20
    # train_image_and_mask_paths = sample_image_and_mask_paths(train_generator, n_sample_images)
    # validation_image_and_mask_paths = sample_image_and_mask_paths(validation_generator, n_sample_images)

    # tensorboard_image_callback = TensorBoardImage(
    #     log_dir=logs_dir.as_posix(),
    #     images_and_masks_paths=train_image_and_mask_paths + validation_image_and_mask_paths)

    csv_logger_callback = CSVLogger(Path(model_dir, 'metrics.csv').as_posix(), append=True)

    results = compiled_model.fit(
        train_generator,
        steps_per_epoch=n_train_steps if generator_type == '3D' else (len(train_generator) if augmentation_type ==
                                                                      'necstlab' else train_config['data_augmentation']['bio_augmentation']['steps_per_epoch']),
        workers=1,
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=n_validation_steps if generator_type == '3D' else (len(
            validation_generator) if augmentation_type == 'necstlab' else train_config['data_augmentation']['bio_augmentation']['validation_steps']),
        # callbacks=[tensorboard_callback]
    )

    metric_names = ['loss'] + [m.name for m in compiled_model.metrics]

    for metric_name in metric_names:

        fig, ax = plt.subplots()
        for split in ['train', 'validate']:

            key_name = metric_name
            if split == 'validate':
                key_name = 'val_' + key_name

            ax.plot(range(epochs), results.history[key_name], label=split)
        ax.set_xlabel('epochs')
        if metric_name == 'loss':
            ax.set_ylabel(compiled_model.loss.__name__)
        else:
            ax.set_ylabel(metric_name)
        ax.legend()
        if metric_name == 'loss':
            fig.savefig(Path(plots_dir, compiled_model.loss.__name__ + '.png').as_posix())
        else:
            fig.savefig(Path(plots_dir, metric_name + '.png').as_posix())

    # mosaic plot
    fig2, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    counter_m = 0
    counter_n = 0
    for metric_name in metric_names:

        for split in ['train', 'validate']:

            key_name = metric_name
            if split == 'validate':
                key_name = 'val_' + key_name

            axes[counter_m, counter_n].plot(range(epochs), results.history[key_name], label=split)
        axes[counter_m, counter_n].set_xlabel('epochs')
        if metric_name == 'loss':
            axes[counter_m, counter_n].set_ylabel(compiled_model.loss.__name__)
        else:
            axes[counter_m, counter_n].set_ylabel(metric_name)
        axes[counter_m, counter_n].legend()

        counter_n += 1
        if counter_n == 3:  # 3 plots per row
            counter_m += 1
            counter_n = 0

    fig2.tight_layout()
    fig2.delaxes(axes[1][2])
    fig2.savefig(Path(plots_dir, 'metrics_mosaic.png').as_posix())

    metadata = {
        'gcp_bucket': gcp_bucket,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'num_classes': 1,
        'target_size': target_size,
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'original_config_filename': config_file,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1),
        'dataset_config': dataset_config,
        'train_config': train_config
    }

    with Path(model_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -r '{}' '{}'".format(Path(tmp_directory, 'models').as_posix(), gcp_bucket))

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--gcp-bucket',
        type=str,
        help='The GCP bucket where the prepared data is located and to use to store the trained model.')
    argparser.add_argument(
        '--config-file',
        type=str,
        help='The location of the train configuration file.')

    train(**argparser.parse_args().__dict__)
