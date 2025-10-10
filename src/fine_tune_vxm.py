# finetune_two_stage.py
# Minimal two-stage fine-tuning for a pretrained VoxelMorph model (TF 2.x)
# Stage 1: freeze early layers; Stage 2: unfreeze all with lower LR

import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from voxelmorph import layers, networks, losses
import voxelmorph as vxm
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint


from evaluation_utils import (
    test_data_generator,
    save_displacement_vector_as_vtk,
    save_as_tiff_uint8,
    binarize_volume,
    compute_diff_map,
    report_combined_difference_percentages,
    
)

from train_utils import (
    initialize_generator_parameters, 
    vxm_data_generator, 
    build_and_train_vxm_model, 
    save_image_as_vtk
)

from cross_validation import (prepare_loocv_fold)




def load_pretrained(path):
    with open(MODEL_JSON, "r") as f:
        arch_json = f.read()
    m = tf.keras.models.model_from_json(arch_json, 
                custom_objects={'SpatialTransformer': vxm.layers.SpatialTransformer, 
                'VxmDense': vxm.networks.VxmDense})
    m.load_weights(path)
    return m


def freeze_first_two_encoder_levels_convs(model):
    """Freeze ONLY Conv3D layers in encoder levels 0 & 1."""
    patterns = ( "vxm_dense_unet_enc_conv_0_", "vxm_dense_unet_enc_conv_1_" )
    frozen, seen = 0, 0
    for layer in model.layers:
        if any(layer.name.startswith(p) for p in patterns):
            seen += 1
            if isinstance(layer, tf.keras.layers.Conv3D):
                layer.trainable = False
                frozen += 1
    print(f"[freeze] froze {frozen} Conv3D layers (encoder levels 0 & 1); "
          f"left {seen - frozen} non-param layers (activations/pooling) untouched.")
    return model



def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    #load model
    model = load_pretrained(MODEL_STAGE1)

    #load data
    folds_to_run = [only_fold] if only_fold is not None else range(num_samples)

    for fold_idx in folds_to_run:
        print(f"========== Starting Fold {fold_idx} ==========")
        train_file, test_file = prepare_loocv_fold(all_data_path, fold_idx, num_samples)

    loss_functions = [losses.NCC().loss, losses.Grad('l2').loss]
    loss_weights = [1, 0.05]

    generator_params = initialize_generator_parameters(hdf5_file=train_file, patch_size=(128, 128, 128))
    print("Initializing VoxelMorph training data generator for first level...")
    train_generator = vxm_data_generator(
       hdf5_file=train_file,
      patch_size=(128, 128, 128),
       batch_size= BATCH_SIZE,
       generator_params=generator_params
    )

    #in_sample, out_sample = next(train_generator)
    """
    # Stage 1: freeze first two levels
    freeze_first_two_encoder_levels_convs(model)

    #compile and fit


    # Compile model with losses and weights
    loss_functions = [losses.NCC().loss, losses.Grad('l2').loss]
    loss_weights = [1, 0.05]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= LR_STAGE1),
                 loss=loss_functions,
                 loss_weights=loss_weights)

    # Define callbacks
    cb_stage1 = [
    CSVLogger(f"{SAVE_DIR}/hist_stage1.csv"),
    ModelCheckpoint(f"{SAVE_DIR}/weights_stage1.h5", save_weights_only=True),
    ReduceLROnPlateau(monitor="loss", factor=0.8, patience=10, min_lr=1e-6, verbose=2),
                ]

    # Train
    start_time = time.time()
    history = model.fit(
            train_generator,
            epochs=EPOCHS_STAGE1,
            steps_per_epoch=STEPS_PER_EPOCH,
            verbose=1,
            callbacks=cb_stage1
        )

    end_time = time.time()
    print(f"✅ Training completed in {(end_time - start_time) / 60:.2f} minutes")
    """
    # Stage 2: unfreeze all with lower LR
    for l in model.layers: l.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= LR_STAGE2),
                  loss=loss_functions,
                  loss_weights=loss_weights)

    cb_stage2 = [
    CSVLogger(f"{SAVE_DIR}/hist_stage2.csv"),
    ModelCheckpoint(f"{SAVE_DIR}/weights_stage2.h5", save_weights_only=True),
    ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=2),
    ]

    history = model.fit(
            train_generator,
            epochs=EPOCHS_STAGE2,
            steps_per_epoch=STEPS_PER_EPOCH,
            verbose=2,
            callbacks=cb_stage2
        )

    # Save final
    with open(f"{SAVE_DIR}/model_finetuned.json","w") as f: f.write(model.to_json())
    model.save_weights(f"{SAVE_DIR}/weights_finetuned.h5")
    print("[done] saved to", SAVE_DIR)



    #testing on the saved model

if __name__=="__main__": 
    # ----------------- USER SETTINGS -----------------
    # Choose ONE way to load the pretrained model:
    MODEL_JSON      = "/home/kchand/results/cross_validation/vxm_model_architecture.json"
    MODEL_WEIGHTS   = "/home/kchand/results/cross_validation/vxm_weights_fold/weights_fold0.h5"
    MODEL_STAGE1    = '/home/kchand/results/finetune_two_stage/weights_stage1.h5'

    TRAIN_HDF5      = "/home/kchand/input_data/all_samples_simple_structures.h5"
    SAVE_DIR        = "/home/kchand/results/finetune_two_stage"

    PATCH_SIZE      = (128, 128, 128)
    BATCH_SIZE      = 8
    STEPS_PER_EPOCH = 100

    # Stage 1 (frozen)
    EPOCHS_STAGE1   = 130


    FREEZE_FIRST_LEVELS = 2      # freeze first N Levels
    LR_STAGE1       = 0.0001


    # Stage 2 (unfrozen)
    EPOCHS_STAGE2   = 50
    LR_STAGE2       = 0.0002
    # -------------------------------------------------

    #data 
    num_samples = 16

    #toggle for a single fold (optional) e.g., set to 0 or 3 to test a single fold; max = num_samples - 1
    only_fold = 15
    all_data_path = '/home/kchand/input_data/all_samples_simple_structures.h5'
    main()
