import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from voxelmorph import layers, networks, losses

from evaluation_utils import (
    test_data_generator,
    save_image_as_vtk,
    save_displacement_vector_as_vtk,
    dice_coefficient,
    save_as_tiff_uint8,
    binarize_volume,
    compute_diff_map,
    report_combined_difference_percentages
)

from train_utils import initialize_generator_parameters, vxm_data_generator, build_and_train_vxm_model

def train_voxelmorph(train_hdf5, save_weights_path, json_path, log_dir=None, nb_epochs=150):
    generator_params = initialize_generator_parameters(hdf5_file=train_hdf5, patch_size=(128, 128, 128))
    train_generator = vxm_data_generator(
        hdf5_file=train_hdf5,
        patch_size=(128, 128, 128),
        batch_size=8,
        generator_params=generator_params
    )

    #with h5py.File(train_hdf5, 'r') as hf:
        #sample_moving = hf['moving_0'][...][np.newaxis, ..., np.newaxis]
        #sample_fixed = hf['static_0'][...][np.newaxis, ..., np.newaxis]

    #in_sample = [sample_moving, sample_fixed]

    in_sample, out_sample = next(train_generator)

    model, history = build_and_train_vxm_model(
        train_generator=train_generator,
        in_sample=in_sample,
        nb_epochs=nb_epochs,
        steps_per_epoch=8
    )

    Path(save_weights_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(save_weights_path)
    # Save the model architecture in JSON format
    
    if not json_path.exists():
        # Save the model architecture in JSON format
        model_json = model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)
        print(f"Model architecture saved to {json_path}")
    else:
        print(f"File already exists: {json_path} — skipping save.")

    if log_dir:
        log_path = os.path.join(log_dir, "training_history.csv")
        pd.DataFrame(history.history).to_csv(log_path, index=False)

def evaluate_voxelmorph(test_hdf5, weights_path, result_dir, json_file):

    with h5py.File(test_hdf5, 'r') as hf:
        moving = hf['moving_0'][...][np.newaxis, ..., np.newaxis]
        fixed = hf['static_0'][...][np.newaxis, ..., np.newaxis]

    vol_shape = moving.shape[1:4]

        # Load model architecture
    with open(json_file, "r") as json_path:
        model_json = json_path.read()

    vxm_model = tf.keras.models.model_from_json(model_json, 
                custom_objects={'SpatialTransformer': vxm.layers.SpatialTransformer, 
                'VxmDense': vxm.networks.VxmDense})

    model.load_weights(weights_path)

    # Initialize the test generator
    test_generator = test_data_generator(test_hdf5, patch_size=(128, 128, 128), stride=(64, 64, 64))
    # Get the output for just one sample
    reconstructed_moved, reconstructed_displacement, fixed_image, moving_image = next(test_generator)
    # Plot the images

    fixed_crop = fixed_image[:, :530, :]
    moving_crop = moving_image[:, :530, :]
    moved_crop = reconstructed_moved[:, :530, :]

    binary_fixed = binarize_volume(fixed_crop)
    binary_moving = binarize_volume(moving_crop)
    binary_moved = binarize_volume(moved_crop)

    dice_before = dice_coefficient(binary_fixed, binary_moving)
    dice_after = dice_coefficient(binary_fixed, binary_moved)

    diff_map_before = compute_diff_map(binary_fixed, binary_moving)
    #return difference map dictionary of BDM
    diff_stats_before = report_combined_difference_percentages(diff_map_before, binary_fixed, binary_moving)

    diff_map_after = compute_diff_map(binary_fixed, binary_moved)
    diff_stats_after = report_combined_difference_percentages(diff_map_after, binary_fixed, binary_moved)

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    save_moved_image_as_vtk(moved, os.path.join(result_dir, "moved_image.vtk"))
    save_moved_image_as_vtk(fixed, os.path.join(result_dir, "fixed_image.vtk"))
    save_moved_image_as_vtk(moving, os.path.join(result_dir, "moving_image.vtk"))
    save_displacement_vector_as_vtk(disp, os.path.join(result_dir, "disp_field.vtk"))
    save_as_tiff(moved, os.path.join(result_dir, "reconstructed_moved.tiff"))
    save_as_tiff(fixed, os.path.join(result_dir, "fixed_image.tiff"))
    save_as_tiff(moving, os.path.join(result_dir, "moving_image.tiff"))

    fold_id = int(Path(result_dir).stem.split('_')[-1])
    results = {
        "fold": fold_id,
        "Dice Before": dice_before,
        "Dice After": dice_after,
        "BDM Before -1 (%)": diff_stats_before["Percent -1"],
        "BDM Before  0 (%)": diff_stats_before["Percent  0"],
        "BDM Before +1 (%)": diff_stats_before["Percent +1"],
        "BDM After  -1 (%)": diff_stats_after["Percent -1"],
        "BDM After   0 (%)": diff_stats_after["Percent  0"],
        "BDM After  +1 (%)": diff_stats_after["Percent +1"]
    }

    pd.DataFrame([results]).to_csv(os.path.join(result_dir, "metrics_fold.csv"), index=False)

def prepare_loocv_fold(input_file, test_idx):
    train_file = '/home/kchand/input_data/train_data_temp.h5'
    test_file = '/home/kchand/input_data/test_data_temp.h5'

    with h5py.File(input_file, 'r') as hf_all:
        with h5py.File(test_file, 'w') as hf_test:
            hf_test.create_dataset('static_0', data=hf_all[f'static_{test_idx}'][...])
            hf_test.create_dataset('moving_0', data=hf_all[f'moving_{test_idx}'][...])
            hf_test['static_0'].attrs['sample_name'] = hf_all[f'static_{test_idx}'].attrs['sample_name']
            hf_test['moving_0'].attrs['sample_name'] = hf_all[f'moving_{test_idx}'].attrs['sample_name']

        with h5py.File(train_file, 'w') as hf_train:
            count = 0
            for i in range(7):
                if i == test_idx:
                    continue
                hf_train.create_dataset(f'static_{count}', data=hf_all[f'static_{i}'][...])
                hf_train.create_dataset(f'moving_{count}', data=hf_all[f'moving_{i}'][...])
                hf_train[f'static_{count}'].attrs['sample_name'] = hf_all[f'static_{i}'].attrs['sample_name']
                hf_train[f'moving_{count}'].attrs['sample_name'] = hf_all[f'moving_{i}'].attrs['sample_name']
                count += 1

    return train_file, test_file

def run_loocv_pipeline():
    all_data_path = '/home/kchand/input_data/all_data_for_cross_validation.h5'
    results_dir = '/home/kchand/results/cross_validation'
    weights_output_dir = os.path.join(results_dir, 'vxm_weights_fold')
    json_path = Path("/home/kchand/results/cross_validation/vxm_model_architecture.json")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(weights_output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for fold_idx in range(7):
        print(f"========== Starting Fold {fold_idx} ==========")
        train_file, test_file = prepare_loocv_fold(all_data_path, fold_idx)
        weights_path = os.path.join(weights_output_dir, f'weights_fold{fold_idx}.h5')
        result_dir = os.path.join(results_dir, f'fold_{fold_idx}')

        train_voxelmorph(train_file, weights_path, json_path)
        evaluate_voxelmorph(test_file, weights_path, result_dir, json_file)

        metrics_csv = os.path.join(result_dir, 'metrics_fold.csv')
        if os.path.exists(metrics_csv):
            df = pd.read_csv(metrics_csv)
            results.append(df.iloc[0].to_dict())

    final_df = pd.DataFrame(results)
    final_df.to_csv(os.path.join(results_dir, 'loocv_all_metrics.csv'), index=False)
    print("\n✅ LOOCV completed. All metrics saved to loocv_all_metrics.csv")

if __name__ == "__main__":
    
    run_loocv_pipeline()
