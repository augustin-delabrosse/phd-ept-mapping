import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold
from osgeo import gdal

import random
import os
import shutil

import numpy as np
import pandas as pd

from codecarbon import EmissionsTracker

from utils import plot_history, custom_train_test_split
from preprocessing import MultimodalDataGenerator, CombinedDataGenerator, get_combined_generator, get_labels
from classification_models import FusionCaiT, get_config
from callbacks import WarmUpCosine

import warnings

warnings.filterwarnings("ignore")


def cross_val_training(model_name, train_path, val_path, test_path, labels_path, epochs, learning_rate, weight_decay, model_output_dir, studied_order, site, campaign, experiment_id, batch_size=4, buffer_size=1000, n_folds=5, split=0.8, save_models=True, image_size=256, train_only=False, ground_truth_training=True):#, only_return_test_dataset=False):
    """Train a model using k-fold cross-validation and optional data augmentation.
    
    Args:
        model_name (str): Name of the model to train.
        img_paths, dsm_paths, dtm_paths, water_paths (list): Paths to data files.
        labels (list): Label data for each image.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        weight_decay (float): Weight decay rate for regularization.
        model_output_dir (str): Directory to save model checkpoints.
        studied_order (str): Data class being studied.
        experiment_id (str): Identifier for the experiment run.
        batch_size (int): Batch size for training and validation.
        buffer_size (int): Buffer size for data shuffling.
        n_folds (int): Number of cross-validation folds.
        split (float): Train/test split ratio.
        save_models (bool): Flag to save trained models.
        image_size (int): Dimension size for input images.
        train_only (bool): Use all samples for training.
        ground_truth_training (bool): Indicate if the training is the training with only ground-truth labels (True) or pseudo labels (False)
        # only_return_test_dataset (bool): Return test_dataset before training.

    Returns:
        if not train_only:
            test_dataset (tf.data.Dataset): Test dataset for further evaluation.
    """
    valid_models = ["fusioncait"]
    if model_name.lower() not in valid_models:
        raise ValueError(f"Invalid model_name '{model_name}'. Choose from {valid_models}.")

    valid_orders = ["ephemeropteres", "plecopeteres", "trichopteres"]
    if studied_order.lower() not in valid_orders:
        raise ValueError(f"Invalid studied_order '{studied_order}'. Choose from {valid_orders}.")

    valid_sites = ["timbertiere"]
    if site.lower() not in valid_sites:
        raise ValueError(f"Invalid site '{site}'. Choose from {valid_sites}.")

    valid_campaigns = ["202404", "202406"]
    if campaign not in valid_campaigns:
        raise ValueError(f"Invalid campaign '{campaign}'. Choose from {valid_campaigns}.")

    train_image_paths = [train_path + "image_patches/" + i for i in os.listdir(train_path + "image_patches/") if i[-3:]=="tif"]
    val_image_paths = [val_path + "image_patches/" + i for i in os.listdir(val_path + "image_patches/") if i[-3:]=="tif"]
    test_image_paths = [test_path + "image_patches/" + i for i in os.listdir(test_path + "image_patches/") if i[-3:]=="tif"]
    
    train_dsm_paths = [train_path + "dsm_patches/" + i for i in os.listdir(train_path + "dsm_patches/") if i[-3:]=="tif"]
    val_dsm_paths = [val_path + "dsm_patches/" + i for i in os.listdir(val_path + "dsm_patches/") if i[-3:]=="tif"]
    test_dsm_paths = [test_path + "dsm_patches/" + i for i in os.listdir(test_path + "dsm_patches/") if i[-3:]=="tif"]
    
    train_dtm_paths = [train_path + "dtm_patches/" + i for i in os.listdir(train_path + "dtm_patches/") if i[-3:]=="tif"]
    val_dtm_paths = [val_path + "dtm_patches/" + i for i in os.listdir(val_path + "dtm_patches/") if i[-3:]=="tif"]
    test_dtm_paths = [test_path + "dtm_patches/" + i for i in os.listdir(test_path + "dtm_patches/") if i[-3:]=="tif"]
    
    train_water_paths = [train_path + "water_patches/" + i for i in os.listdir(train_path + "water_patches/") if i[-3:]=="tif"]
    val_water_paths = [val_path + "water_patches/" + i for i in os.listdir(val_path + "water_patches/") if i[-3:]=="tif"]
    test_water_paths = [test_path + "water_patches/" + i for i in os.listdir(test_path + "water_patches/") if i[-3:]=="tif"]
    
    df_labels = pd.read_csv(labels_path, index_col="Unnamed: 0")
    train_labels = get_labels(df_labels, train_image_paths, studied_order, site.capitalize(), campaign)
    val_labels = get_labels(df_labels, val_image_paths, studied_order, site.capitalize(), campaign)
    test_labels = get_labels(df_labels, test_image_paths, studied_order, site.capitalize(), campaign)
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_folds) 

    if train_only:
        train_image_paths = train_image_paths + val_image_paths + test_image_paths
        train_dsm_paths = train_dsm_paths + val_dsm_paths + test_dsm_paths
        train_dtm_paths = train_dtm_paths + val_dtm_paths + test_dtm_paths
        train_water_paths = train_water_paths + val_water_paths + test_water_paths
        train_labels = train_labels + val_labels + test_labels
        train_indices = random.sample(range(len(train_image_paths)), len(train_image_paths))
    
    else:
        test_gen = MultimodalDataGenerator(
            img_paths=test_image_paths,
            dsm_paths=test_dsm_paths,
            dtm_paths=test_dtm_paths,
            water_paths=test_water_paths,
            labels=test_labels,
            batch_size=batch_size,
            cutmix_fusion=False,
            augment=False,
            pad=False,
            img_size=256
        )    

        train_image_paths += val_image_paths
        train_dsm_paths += val_dsm_paths
        train_dtm_paths += val_dtm_paths
        train_water_paths += val_water_paths
        train_labels += val_labels
        train_indices = random.sample(range(len(train_image_paths)), len(train_image_paths))
    
        # if only_return_test_dataset:
        #     return test_dataset
    
    # Initialize emissions tracker for tracking environmental impact
    tracker = EmissionsTracker(experiment_id=str(experiment_id), measure_power_secs=60, allow_multiple_runs=True)
    tracker.start()

    if train_only:
        prop = round(np.sum(np.array(train_labels) > 0)/len(train_labels), 3)
    else:
        prop = round(np.sum(np.array(train_labels + test_labels) > 0)/len(train_labels + test_labels), 3)
    print(f"proportion of {studied_order} in full dataset: {prop}%")
    
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    
    fold = 1

    # Perform k-fold training and validation
    for train_index, val_index in kf.split(np.zeros(len(train_indices)),train_indices):
        print(f'================================= FOLD {fold}/{n_folds} ==============================================')

        
        # Split data paths and labels for each fold
        fold_train_indices = pd.Series(train_indices)[train_index].values
        fold_val_indices = pd.Series(train_indices)[val_index].values

        # print(len(fold_train_indices), len(train_indices), len(train_index), len(train_image_paths))
        
        train_image_paths_ = [train_image_paths[i] for i in fold_train_indices]
        train_dsm_paths_ = [train_dsm_paths[i] for i in fold_train_indices]
        train_dtm_paths_ = [train_dtm_paths[i] for i in fold_train_indices]
        train_water_paths_ = [train_water_paths[i] for i in fold_train_indices]
        train_labels_ = [train_labels[i] for i in fold_train_indices]
        
        val_image_paths = [train_image_paths[i] for i in fold_val_indices]
        val_dsm_paths = [train_dsm_paths[i] for i in fold_val_indices]
        val_dtm_paths = [train_dtm_paths[i] for i in fold_val_indices]
        val_water_paths = [train_water_paths[i] for i in fold_val_indices]
        val_labels = [train_labels[i] for i in fold_val_indices]

        # Check class distribution for current fold
        print(f"proportion of {studied_order} in train dataset: {round(np.sum(np.array(train_labels_) > 0)/len(train_labels_), 3)}%")
        print(f"proportion of {studied_order} in val dataset: {round(np.sum(np.array(val_labels) > 0)/len(val_labels), 3)}%")
        print('')
        
        train_gen = get_combined_generator(
            img_paths=train_image_paths_,
            dsm_paths=train_dsm_paths_,
            dtm_paths=train_dtm_paths_,
            water_paths=train_water_paths_,
            labels=train_labels_,
            batch_size=batch_size,
            cutmix_fusion=False,
            augment_first_gen=True,
            augment_second_gen=False,
            cutmix_second_gen=False,
            mixup_first_gen=True,
            mixup_second_gen=True,
            img_size=256
        )
                
        val_gen = MultimodalDataGenerator(
            img_paths=val_image_paths,
            dsm_paths=val_dsm_paths,
            dtm_paths=val_dtm_paths,
            water_paths=val_water_paths,
            labels=val_labels,
            batch_size=batch_size,
            cutmix_fusion=False,
            augment=False,
            use_mixup=False,
            pad=False,
            img_size=256
        )

        # Initialize model if specified
        if str(model_name).lower() == "fusioncait":
            config = get_config()
            model = FusionCaiT(**config)

        # Configure learning rate schedule and optimizer
        total_steps = int((len(train_image_paths_) / batch_size) * epochs)
        warmup_epoch_percentage = 0.10 # Warm-up for 10% of total steps
        warmup_steps = int(total_steps * warmup_epoch_percentage)
        scheduled_lrs = WarmUpCosine(
            learning_rate_base=learning_rate,
            total_steps=total_steps,
            warmup_learning_rate=0.0,
            warmup_steps=warmup_steps,
        )
        
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                # keras.metrics.SparseTopKCategoricalAccuracy(2, name="top-2-accuracy"),
            ],
        )
    
        callbacks = [
            tf.keras.callbacks.EarlyStopping(start_from_epoch=np.max([20, epochs//10]), patience=np.max([20, epochs//10]), restore_best_weights=True),
                ]

        # Train model and plot results for each fold
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

        plot_history(history.history['accuracy'], history.history['val_accuracy'],
                     history.history['loss'], history.history['val_loss'])

        # Save model if specified
        if save_models:
            os.makedirs(model_output_dir, exist_ok=True)
            if ground_truth_training:
                saving_path = model_output_dir + 'ground_truth_training/'
            else:
                saving_path = model_output_dir + 'pseudo_labels_training/'

            os.makedirs(saving_path, exist_ok=True)
            os.makedirs(saving_path + f'fold_{fold}/', exist_ok=True)
            model.save(saving_path + f'fold_{fold}/' + f'model_weights.ckpt')

        loss, accuracy = model.evaluate(val_gen)
        print(f"Val accuracy: {round(accuracy * 100, 2)}%")
        print(f"Val loss: {round(loss, 6)}")

        # Gather results
        VALIDATION_ACCURACY.append(accuracy)
        VALIDATION_LOSS.append(loss)
    
        fold += 1
    
        tf.keras.backend.clear_session()
    
    
    print('''================================ END OF TRAINING =========================================''')
    print(f'''
    VALIDATION ACCURACIES: {VALIDATION_ACCURACY}''')
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions}")

    if not train_only:
        return test_gen
    else:
        return None

        

def load_best_model(model_folder, dataset, n_folds=5):
    """
    Load the best model from a series of cross-validation folds based on evaluation loss.
    
    Args:
        model_folder (str): Path to the directory containing model weights for each fold.
        dataset (tf.data.Dataset): Dataset for evaluating each fold's performance.
        n_folds (int): Number of folds used in cross-validation.
        
    Returns:
        model (tf.keras.Model): The model with the best performance (lowest loss) on the dataset.
    """
    # Initialize variables to track the best model and its associated fold
    best_loss = 1000000
    best_model = None
    best_fold = None

    # Loop through each fold to load, evaluate, and identify the best-performing model
    for i in range(n_folds):
        # Construct path for model weights for the current fold
        model_path = os.path.join(model_folder, f"fold_{i+1}", "model_weights.ckpt")

        # Load the model weights for the current fold
        model = tf.keras.models.load_model(model_path)

        # Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(dataset)
        print(f"fold_{i+1} accuracy: {round(accuracy*100, 2)}%")
        print(f"fold_{i+1} loss: {round(loss, 5)}")

        # Update the best model if current fold has a lower loss
        if loss < best_loss:
            best_loss = loss
            best_model = model_path
            best_fold = i+1
            
        tf.keras.backend.clear_session()

    print(f"Loading best model (fold n°{best_fold})...")

    # Load and return the best model
    model = tf.keras.models.load_model(best_model)
    return model


def simple_training(model_name, 
                     train_path, 
                    val_path,
                     test_path, 
                     labels_path, 
                     epochs, 
                     learning_rate, 
                     weight_decay, 
                     model_output_dir, 
                     studied_order, 
                     site,
                     campaign,
                     experiment_id, 
                     batch_size=4, 
                     buffer_size=1000, 
                     split=0.75, 
                     save_models=True, 
                     image_size=256, 
                     train_only=False, 
                     ground_truth_training=True):
                     # only_return_test_dataset=False, ):
    """
    Train a specified model on the given datasets with options for dataset splitting, data augmentation,
    and environmental impact tracking.

    Args:
        model_name (str): Name of the model to train.
        train_path, val_path, test_path (list): paths to train, val and test data.
        label_path (str): path to labels.
        labels (list): List of labels associated with each sample.
        epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for the optimizer.
        model_output_dir (str): Directory to save the trained model.
        studied_order (str): Name of the order being studied for label calculation.
        experiment_id (str): ID for tracking emissions.
        batch_size (int): Batch size for training.
        buffer_size (int): Buffer size for shuffling.
        split (float): Fraction of data to use for training (the rest for testing/validation).
        save_models (bool): Whether to save the model after training.
        image_size (int): Image dimensions to resize to.
        train_only (bool): Whether to train without validation set.
        ground_truth_training (bool): Indicates ground truth training for file saving.
        # only_return_test_dataset (bool): If True, only returns the test dataset.

    Returns:
        model: The trained model.
        train_gen, val_gen, test_gen: Sequences for training, validation, and testing.

    Raises:
        ValueError: If `model_name` is invalid.
    """
    valid_models = ["fusioncait"]
    if model_name.lower() not in valid_models:
        raise ValueError(f"Invalid model_name '{model_name}'. Choose from {valid_models}.")

    valid_orders = ["ephemeropteres", "plecopeteres", "trichopteres"]
    if studied_order.lower() not in valid_orders:
        raise ValueError(f"Invalid studied_order '{studied_order}'. Choose from {valid_orders}.")

    valid_sites = ["timbertiere"]
    if site.lower() not in valid_sites:
        raise ValueError(f"Invalid site '{site}'. Choose from {valid_sites}.")

    valid_campaigns = ["202404", "202406"]
    if campaign not in valid_campaigns:
        raise ValueError(f"Invalid campaign '{campaign}'. Choose from {valid_campaigns}.")

    train_image_paths = [train_path + "image_patches/" + i for i in os.listdir(train_path + "image_patches/") if i[-3:]=="tif"]
    val_image_paths = [val_path + "image_patches/" + i for i in os.listdir(val_path + "image_patches/") if i[-3:]=="tif"]
    test_image_paths = [test_path + "image_patches/" + i for i in os.listdir(test_path + "image_patches/") if i[-3:]=="tif"]
    
    train_dsm_paths = [train_path + "dsm_patches/" + i for i in os.listdir(train_path + "dsm_patches/") if i[-3:]=="tif"]
    val_dsm_paths = [val_path + "dsm_patches/" + i for i in os.listdir(val_path + "dsm_patches/") if i[-3:]=="tif"]
    test_dsm_paths = [test_path + "dsm_patches/" + i for i in os.listdir(test_path + "dsm_patches/") if i[-3:]=="tif"]
    
    train_dtm_paths = [train_path + "dtm_patches/" + i for i in os.listdir(train_path + "dtm_patches/") if i[-3:]=="tif"]
    val_dtm_paths = [val_path + "dtm_patches/" + i for i in os.listdir(val_path + "dtm_patches/") if i[-3:]=="tif"]
    test_dtm_paths = [test_path + "dtm_patches/" + i for i in os.listdir(test_path + "dtm_patches/") if i[-3:]=="tif"]
    
    train_water_paths = [train_path + "water_patches/" + i for i in os.listdir(train_path + "water_patches/") if i[-3:]=="tif"]
    val_water_paths = [val_path + "water_patches/" + i for i in os.listdir(val_path + "water_patches/") if i[-3:]=="tif"]
    test_water_paths = [test_path + "water_patches/" + i for i in os.listdir(test_path + "water_patches/") if i[-3:]=="tif"]
    
    df_labels = pd.read_csv(labels_path, index_col="Unnamed: 0")
    train_labels = get_labels(df_labels, train_image_paths, studied_order, site.capitalize(), campaign)
    val_labels = get_labels(df_labels, val_image_paths, studied_order, site.capitalize(), campaign)
    test_labels = get_labels(df_labels, test_image_paths, studied_order, site.capitalize(), campaign)

    if train_only:
        # Data summary
        print(f"proportion of {studied_order} in whole dataset: {round(np.sum(np.array(train_labels + test_labels) > 0)/len(train_labels + test_labels), 3)}% ({len(train_labels + test_labels)} samples)")
        print(f"proportion of {studied_order} in train dataset: {round(np.sum(np.array(train_labels) > 0)/len(train_labels), 3)}% ({len(train_labels)} samples)")
        print(f"proportion of {studied_order} in test dataset: {round(np.sum(np.array(test_labels) > 0)/len(test_labels), 3)}% ({len(test_labels)} samples)")
        
        test_gen = MultimodalDataGenerator(
            img_paths=test_image_paths,
            dsm_paths=test_dsm_paths,
            dtm_paths=test_dtm_paths,
            water_paths=test_water_paths,
            labels=test_labels,
            batch_size=batch_size,
            cutmix_fusion=False,
            augment=False,
            pad=False,
            img_size=256
        )
        
        # if only_return_test_dataset:
        #     return test_dataset
        
        train_image_paths += val_image_paths
        train_dsm_paths += val_dsm_paths
        train_dtm_paths += val_dtm_paths
        train_water_paths += val_water_paths
        train_labels += val_labels
        
        train_gen = get_combined_generator(
            img_paths=train_image_paths,
            dsm_paths=train_dsm_paths,
            dtm_paths=train_dtm_paths,
            water_paths=train_water_paths,
            labels=train_labels,
            batch_size=batch_size,
            cutmix_fusion=False,
            augment_first_gen=True,
            augment_second_gen=False,
            cutmix_second_gen=False,
            mixup_first_gen=True,
            mixup_second_gen=True,
            img_size=256
        )

    else:   
        # Dataset proportions summary
        print(f"proportion of {studied_order} in whole dataset: {round(np.sum(np.array(train_labels + val_labels + test_labels) > 0)/len(train_labels + val_labels + test_labels), 3)}% ({len(train_labels + val_labels + test_labels)} samples)")
        print(f"proportion of {studied_order} in train dataset: {round(np.sum(np.array(train_labels) > 0)/len(train_labels), 3)}% ({len(train_labels)} samples)")
        print(f"proportion of {studied_order} in val dataset: {round(np.sum(np.array(val_labels) > 0)/len(val_labels), 3)}% ({len(val_labels)} samples)")
        print(f"proportion of {studied_order} in test dataset: {round(np.sum(np.array(test_labels) > 0)/len(test_labels), 3)}% ({len(test_labels)} samples)")
    
        train_gen = get_combined_generator(
            img_paths=train_image_paths,
            dsm_paths=train_dsm_paths,
            dtm_paths=train_dtm_paths,
            water_paths=train_water_paths,
            labels=train_labels,
            batch_size=batch_size,
            cutmix_fusion=False,
            augment_first_gen=True,
            augment_second_gen=False,
            cutmix_second_gen=False,
            mixup_first_gen=True,
            mixup_second_gen=True,
            img_size=256
        )
                
        val_gen = MultimodalDataGenerator(
            img_paths=val_image_paths,
            dsm_paths=val_dsm_paths,
            dtm_paths=val_dtm_paths,
            water_paths=val_water_paths,
            labels=val_labels,
            batch_size=batch_size,
            cutmix_fusion=False,
            augment=False,
            use_mixup=False,
            pad=False,
            img_size=256
        )
        
        test_gen = MultimodalDataGenerator(
            img_paths=test_image_paths,
            dsm_paths=test_dsm_paths,
            dtm_paths=test_dtm_paths,
            water_paths=test_water_paths,
            labels=test_labels,
            batch_size=batch_size,
            cutmix_fusion=False,
            augment=False,
            pad=False,
            img_size=256
        )
        
        # if only_return_test_dataset:
        #     return test_dataset

    # Initialize emissions tracker for tracking environmental impact
    tracker = EmissionsTracker(experiment_id=str(experiment_id), measure_power_secs=60, allow_multiple_runs=True)
    tracker.start()
    
    # Initialize model if specified
    if str(model_name).lower() == "fusioncait":
        config = get_config()
        model = FusionCaiT(**config)

    # Configure learning rate schedule and optimizer
    total_steps = int((len(train_labels) / batch_size) * epochs)
    warmup_epoch_percentage = 0.10 # Warm-up for 10% of total steps
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=learning_rate,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )
    
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            # keras.metrics.SparseTopKCategoricalAccuracy(2, name="top-2-accuracy"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(start_from_epoch=np.max([20, epochs//10]), patience=np.max([20, epochs//10]), restore_best_weights=True),
            ]

    # Model training
    if train_only:
        history = model.fit(train_gen, validation_data=test_gen, epochs=epochs, callbacks=callbacks)
    else:
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
        
    # Plot training history
    plot_history(history.history['accuracy'], history.history['val_accuracy'],
                 history.history['loss'], history.history['val_loss'])

    # Save model if specified
    if save_models:
        os.makedirs(model_output_dir, exist_ok=True)
        if ground_truth_training:
            saving_path = model_output_dir + 'ground_truth_training/'
        else:
            saving_path = model_output_dir + 'pseudo_labels_training/'
        
        os.makedirs(saving_path, exist_ok=True)
        model.save(saving_path + f'model_weights.ckpt')

        print(f"model saved (path: {saving_path + f'model_weights.ckpt'}")
            

    if not train_only:
        val_loss, val_accuracy = model.evaluate(val_gen)
        print(f"Val accuracy: {round(val_accuracy * 100, 2)}%")
        print(f"Val loss: {round(val_loss, 6)}")

    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"{'Test' if not train_only else 'Val'} accuracy: {round(test_accuracy * 100, 2)}%")
    print(f"{'Test' if not train_only else 'Val'} loss: {round(test_loss, 6)}")

    tf.keras.backend.clear_session()

    emissions: float = tracker.stop()
    print(f"Emissions: {emissions}")

    return (model, train_gen, test_gen) if train_only else (model, train_gen, val_gen, test_gen)


def train(model_name,
          train_path,
          val_path,
          test_path,
          labels_path,
         epochs, 
         learning_rate, 
         weight_decay, 
         model_output_dir, 
         studied_order,
          site,
          campaign,
         experiment_id, 
          cross_val=False,
          n_folds=5,
         batch_size=4, 
         buffer_size=1000, 
         split=0.75, 
         save_models=True, 
         image_size=256, 
         train_only=False,  
         ground_truth_training=True):
    """
    Train a model with optional cross-validation or simple training.

    Parameters:
    - model_name (str): Name of the model architecture to use.
    - train_path, val_path, test_path, labels_path (list): Lists of file paths for the train, val, test and labels inputs.
    - epochs (int): Number of training epochs.
    - learning_rate (float): Initial learning rate for the optimizer.
    - weight_decay (float): Weight decay (L2 penalty) for the optimizer.
    - model_output_dir (str): Directory where the model should be saved.
    - studied_order (str): The specific target label or feature being studied.
    - experiment_id (str): Unique identifier for tracking the experiment.
    - cross_val (bool): If True, perform cross-validation. Otherwise, use a single train-test split.
    - n_folds (int): Number of folds for cross-validation. Used only if cross_val is True.
    - batch_size (int): Number of samples per batch.
    - buffer_size (int): Buffer size for shuffling the dataset.
    - split (float): Proportion of data for training (remainder used for testing).
    - save_models (bool): Whether to save the model weights after training.
    - image_size (int): Dimension for resizing input images.
    - train_only (bool): If True, train only and skip validation.
    - ground_truth_training (bool): If True, trains on ground truth labels; otherwise, may use pseudo-labeling.

    Returns:
    - model: Trained model instance.
    """
    
    if cross_val:
        test_gen = cross_val_training(model_name=model_name,
                            train_path=train_path,
                            val_path=val_path,
                            test_path=test_path,
                            labels_path=labels_path,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            model_output_dir=model_output_dir, 
                            studied_order=studied_order,
                            site=site,
                            campaign=campaign,
                            experiment_id=experiment_id,
                            batch_size=batch_size,
                            buffer_size=buffer_size,
                            n_folds=n_folds,
                            split=split,
                            save_models=True,
                            image_size=image_size,
                            train_only=train_only,
                            ground_truth_training=ground_truth_training)
        
        if test_gen is None:
            train_image_paths = [train_path + "image_patches/" + i for i in os.listdir(train_path + "image_patches/") if i[-3:]=="tif"]
            val_image_paths = [val_path + "image_patches/" + i for i in os.listdir(val_path + "image_patches/") if i[-3:]=="tif"]
            test_image_paths = [test_path + "image_patches/" + i for i in os.listdir(test_path + "image_patches/") if i[-3:]=="tif"]
            image_paths = train_image_paths + val_image_paths + test_image_paths
            
            train_dsm_paths = [train_path + "dsm_patches/" + i for i in os.listdir(train_path + "dsm_patches/") if i[-3:]=="tif"]
            val_dsm_paths = [val_path + "dsm_patches/" + i for i in os.listdir(val_path + "dsm_patches/") if i[-3:]=="tif"]
            test_dsm_paths = [test_path + "dsm_patches/" + i for i in os.listdir(test_path + "dsm_patches/") if i[-3:]=="tif"]
            dsm_paths = train_dsm_paths + val_dsm_paths + test_dsm_paths
            
            train_dtm_paths = [train_path + "dtm_patches/" + i for i in os.listdir(train_path + "dtm_patches/") if i[-3:]=="tif"]
            val_dtm_paths = [val_path + "dtm_patches/" + i for i in os.listdir(val_path + "dtm_patches/") if i[-3:]=="tif"]
            test_dtm_paths = [test_path + "dtm_patches/" + i for i in os.listdir(test_path + "dtm_patches/") if i[-3:]=="tif"]
            dtm_paths = train_dtm_paths + val_dtm_paths + test_dtm_paths
            
            train_water_paths = [train_path + "water_patches/" + i for i in os.listdir(train_path + "water_patches/") if i[-3:]=="tif"]
            val_water_paths = [val_path + "water_patches/" + i for i in os.listdir(val_path + "water_patches/") if i[-3:]=="tif"]
            test_water_paths = [test_path + "water_patches/" + i for i in os.listdir(test_path + "water_patches/") if i[-3:]=="tif"]
            water_paths = train_water_paths + val_water_paths + test_water_paths
            
            df_labels = pd.read_csv(labels_path, index_col="Unnamed: 0")
            train_labels = get_labels(df_labels, train_image_paths, studied_order, site.capitalize(), campaign)
            val_labels = get_labels(df_labels, val_image_paths, studied_order, site.capitalize(), campaign)
            test_labels = get_labels(df_labels, test_image_paths, studied_order, site.capitalize(), campaign)
            labels = train_labels + val_labels + test_labels
            
            test_gen = MultimodalDataGenerator(
                img_paths=image_paths,
                dsm_paths=dsm_paths,
                dtm_paths=dtm_paths,
                water_paths=water_paths,
                labels=labels,
                batch_size=batch_size,
                cutmix_fusion=False,
                augment=False,
                pad=False,
                img_size=256
            )

        # if save_models:
        path = model_output_dir + 'ground_truth_training/' if ground_truth_training else model_output_dir + 'pseudo_labels_training/'
        model = load_best_model(model_folder=path, dataset=test_gen, n_folds=n_folds)
        if not save_models:
            print('Deleting unwanted saved models...')
            shutil.rmtree(path)
    else:
        output = simple_training(model_name=model_name,
                                 train_path=train_path,
                                 val_path=val_path,
                                 test_path=test_path,
                                 labels_path=labels_path,
                                 epochs=epochs,
                                 learning_rate=learning_rate,
                                 weight_decay=weight_decay,
                                 model_output_dir=model_output_dir, 
                                 studied_order=studied_order,
                                 site=site,
                                 campaign=campaign,
                                 experiment_id=experiment_id,
                                 batch_size=batch_size,
                                 buffer_size=buffer_size,
                                 split=split,
                                 save_models=save_models,
                                 image_size=image_size,
                                 train_only=train_only,
                                 ground_truth_training=ground_truth_training)

        if train_only:
            model, train_gen, val_gen = output
        else:
            model, train_gen, val_gen, test_gen = output
            
    return model



def predict_pseudo_labels(model, img_paths, dsm_paths, dtm_paths, water_paths, save_pseudo_labels=True):
    """
    Predict pseudo-labels for a dataset using a provided model and save them to a CSV file if required.
    
    Args:
        model (tf.keras.Model): Trained model used to generate predictions.
        img_paths (list): List of file paths to the image inputs.
        dsm_paths (list): List of file paths to DSM inputs.
        dtm_paths (list): List of file paths to DTM inputs.
        water_paths (list): List of file paths to water data inputs.
        save_pseudo_labels (bool): Whether to save the generated pseudo-labels to a CSV file. Defaults to True.
        
    Returns:
        pseudo_labels (pd.DataFrame): DataFrame containing image paths and their associated pseudo-labels.
    """
    # Ensure all input lists are of the same length
    assert (len(img_paths) == len(dsm_paths)) and (len(img_paths) == len(dtm_paths)) and (len(img_paths) == len(water_paths))
    
    # Create a TensorFlow dataset from the input file paths, with dummy labels
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, dsm_paths, dtm_paths, water_paths, np.zeros(len(img_paths))))
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(4)
    
    # Predict pseudo-labels using the model
    preds = model.predict(dataset)
    
    # Load or initialize the pseudo_labels DataFrame to store predictions
    if os.path.exists("pseudo_labels.csv"):
        pseudo_labels = pd.read_csv('pseudo_labels.csv')
    else:
        pseudo_labels = pd.DataFrame(columns=['img_path', 'dsm_path', 'dtm_path', 'water_path', 'pseudo_label', "rounded_pseudo_label"])
    
    # Append new pseudo-labels to the DataFrame
    for img_path, dsm_path, dtm_path, water_path, prediction in zip(img_paths, dsm_paths, dtm_paths, water_paths, [pred[0] for pred in preds]):
        pseudo_labels.loc[pseudo_labels.shape[0]] = [img_path, dsm_path, dtm_path, water_path, prediction, round(prediction)]
    
    # Save the updated DataFrame to a CSV file if save_pseudo_labels is True
    if save_pseudo_labels:
        pseudo_labels.to_csv('pseudo_labels.csv')
        
    return pseudo_labels
