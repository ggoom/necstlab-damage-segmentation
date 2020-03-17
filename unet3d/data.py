import os

import numpy as np
import tables
import sys

from .normalize import normalize_data_storage, reslice_image_set, read_image_files

import h5py
import sys


def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    # sys.stdout.write("*************" + str(data_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    # affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
    #                                          filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels,
                             truth_dtype=np.uint8, crop=True):
    for set_of_files in image_files:
        images = read_image_files(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
        subject_data = images  # [image  for image in images]
        add_data_to_storage(data_storage, truth_storage, subject_data, n_channels,
                            truth_dtype)
    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, subject_data, n_channels, truth_dtype):
    # sys.stdout.write(str(np.asarray(subject_data[:n_channels])[np.newaxis].shape))
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    # affine_storage.append(np.asarray(affine)[np.newaxis])


def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image.
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'),
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it.
    """
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0]) - 1

    try:
        hdf5_file, data_storage, truth_storage = create_data_file(out_file,
                                                                  n_channels=n_channels,
                                                                  n_samples=n_samples,
                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                             truth_dtype=truth_dtype, n_channels=n_channels, crop=crop)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()

    with h5py.File(out_file, 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        # Get the data
        data = list(f[a_group_key])
        sys.stdout.write(str(data))

    return out_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
