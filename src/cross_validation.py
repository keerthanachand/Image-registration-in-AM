import os
import h5py
import numpy as np
from tqdm import tqdm
import subprocess
import pandas as pd

# train_voxelmorph_lib.py
import json
import tensorflow as tf
import voxelmorph as vxm
import numpy as np
import h5py
import os
from pathlib import Path

def train_voxelmorph(train_hdf5, architecture_path, save_weights_path, log_dir=None):
    # Load architecture
    with open(architecture_path, "r") as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(
        model_json,
        custom_objects={
            'SpatialTransformer': vxm.layers.SpatialTransformer,
            'VxmDense': vxm.networks.VxmDense
        }
    )

    # Compile model
    model.compile(optimizer='adam', loss=[vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss])

    # Load data
    with h5py.File(train_hdf5, 'r') as hf:
        moving = np.stack([hf[key][...] for key in hf if key.startswith('moving_')])
        fixed = np.stack([hf[key][...] for key in hf if key.startswith('static_')])

    moving = moving[..., np.newaxis]  # add channel dim
    fixed = fixed[..., np.newaxis]

    # Train
    history = model.fit([moving, fixed], [fixed, np.zeros_like(moving)], batch_size=1, epochs=10, verbose=1)

    # Save weights
    Path(save_weights_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(save_weights_path)
    print(f"✅ Model weights saved to {save_weights_path}")

    # Optionally save history
    if log_dir:
        log_path = os.path.join(log_dir, "training_history.csv")
        import pandas as pd
        pd.DataFrame(history.history).to_csv(log_path, index=False)
        print(f"📄 Training history saved to {log_path}")

def evaluate_voxelmorph(test_hdf5, architecture_path, weights_path, result_dir):
    from test_utils import test_data_generator, save_moved_image_as_vtk, save_displacement_vector_as_vtk, dice_coefficient, save_as_tiff
    import json
    import pyvista as pv
    import pandas as pd

    # Load model architecture
    with open(architecture_path, "r") as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(
        model_json,
        custom_objects={
            'SpatialTransformer': vxm.layers.SpatialTransformer,
            'VxmDense': vxm.networks.VxmDense
        }
    )
    model.load_weights(weights_path)

    # Run prediction
    test_gen = test_data_generator(test_hdf5, patch_size=(128, 128, 128), stride=(64, 64, 64))
    moved, disp, fixed, moving = next(test_gen)

    # Compute Dice score
    dice_before = dice_coefficient(fixed[:,:530,:], moving[:,:530,:])
    dice_after = dice_coefficient(fixed[:,:530,:], moved[:,:530,:])

    # Save outputs
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    save_moved_image_as_vtk(moved, os.path.join(result_dir, "moved_image.vtk"))
    save_moved_image_as_vtk(fixed, os.path.join(result_dir, "fixed_image.vtk"))
    save_moved_image_as_vtk(moving, os.path.join(result_dir, "moving_image.vtk"))
    save_displacement_vector_as_vtk(disp, os.path.join(result_dir, "disp_field.vtk"))

    save_as_tiff(moved, os.path.join(result_dir, "reconstructed_moved.tiff"))
    save_as_tiff(fixed, os.path.join(result_dir, "fixed_image.tiff"))
    save_as_tiff(moving, os.path.join(result_dir, "moving_image.tiff"))

    # Save Dice scores
    pd.DataFrame({"Dice Before": [dice_before], "Dice After": [dice_after]}).to_csv(
        os.path.join(result_dir, "dice_scores.csv"), index=False
    )
    print(f"🎯 Dice before: {dice_before:.4f}, after: {dice_after:.4f}")


# File paths
all_data_path = '/home/kchand/input_data/all_data_for_cross_validation.h5'
results_dir = '/home/kchand/results/cross_validation'
os.makedirs(results_dir, exist_ok=True)

results_csv = os.path.join(results_dir, 'loocv_metrics.csv')
model_architecture = '/home/kchand/results/vxm_model_architecture.json'
weights_output_dir = os.path.join(results_dir, 'vxm_weights_fold')
os.makedirs(weights_output_dir, exist_ok=True)

def train_model_for_fold(train_data_path, fold_idx):
    weight_path = os.path.join(weights_output_dir, f'weights_fold{fold_idx}.h5')
    cmd = [
        'python', 'train_voxelmorph.py',
        '--train_data', train_data_path,
        '--model_json', model_architecture,
        '--save_weights', weight_path
    ]
    subprocess.run(cmd)
    return weight_path

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

    print(f"[Fold {test_idx}] Prepared LOOCV train/test data")
    return train_file, test_file

def evaluate_model(fold_idx, weights_path, test_file):
    results_subdir = os.path.join(results_dir, f'fold_{fold_idx}')
    os.makedirs(results_subdir, exist_ok=True)
    cmd = [
        'python', 'evaluate_voxelmorph.py',
        '--test_data', test_file,
        '--weights', weights_path,
        '--fold', str(fold_idx),
        '--output_dir', results_subdir
    ]
    subprocess.run(cmd)

results = []
for i in range(7):
    print(f"========== Starting Fold {i} ==========")
    train_file, test_file = prepare_loocv_fold(all_data_path, test_idx=i)
    weights_file = train_model_for_fold(train_file, i)
    evaluate_model(i, weights_file, test_file)
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        if 'fold' in df.columns and i in df['fold'].values:
            results.append(df[df['fold'] == i].iloc[0].to_dict())

final_df = pd.DataFrame(results)
final_df.to_csv(os.path.join(results_dir, 'loocv_all_metrics.csv'), index=False)
print("LOOCV completed. All metrics saved to loocv_all_metrics.csv")