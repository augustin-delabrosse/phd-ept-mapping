from osgeo import gdal

import tensorflow as tf
from tensorflow.keras.utils import Sequence
import tensorflow_addons as tfa
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import cv2

import random
import warnings

warnings.filterwarnings("ignore")

def get_labels(df_labels, img_paths, order, site, campaign):
    """
    Retrieves labels from a DataFrame based on specified image paths, order, site, and campaign.
    
    Args:
        df_labels (pd.DataFrame): DataFrame containing columns 'id', 'site', 'campaign', and label columns for each order.
        img_paths (list): List of image file paths used to extract IDs.
        order (str): Specifies the insect order for label retrieval (options: 'Ephemeropteres', 'Plecopeteres', 'Trichopteres').
        site (str): Specifies the site (options: 'Timbertiere', 'Cisse', 'Fao', 'Roudoudour', 'Louroux').
        campaign (str): Specifies the campaign (options: '202404', '202406').

    Returns:
        list: List of labels corresponding to the specified order, site, and campaign.

    Raises:
        ValueError: If `order`, `site`, or `campaign` is not among the allowed options.
    """

    # Validate the parameters
    valid_orders = ['Ephemeropteres', 'Plecopeteres', 'Trichopteres']
    valid_sites = ['Timbertiere', 'Cisse', 'Fao', 'Roudoudour', 'Louroux']
    valid_campaigns = ['202404', '202406']

    if order not in valid_orders:
        raise ValueError(f"Invalid order '{order}'. Must be one of {valid_orders}.")
    if site not in valid_sites:
        raise ValueError(f"Invalid site '{site}'. Must be one of {valid_sites}.")
    if campaign not in valid_campaigns:
        raise ValueError(f"Invalid campaign '{campaign}'. Must be one of {valid_campaigns}.")

    # Extract IDs from image paths for matching
    ids_order = [int(i.split('_')[-1].split('.')[0]) for i in img_paths]
    sites_order = [str(site) for _ in range(len(img_paths))]
    campaigns_order = [float(campaign) for _ in range(len(img_paths))]

    # Create a DataFrame to facilitate merging and sorting
    order_df = pd.DataFrame({
        'id': ids_order,
        'site': sites_order,
        'campaign': campaigns_order
    })

    # Merge with the main labels DataFrame to match IDs, sites, and campaigns
    df_ordered = pd.merge(order_df, df_labels, on=['id', 'site', 'campaign'], how='left')
    df_ordered.drop_duplicates(inplace=True)

    # Extract the labels for the specified order
    labels = df_ordered[order].values.tolist()

    return labels


class MultimodalDataGenerator(Sequence):
    """
    Data generator for multimodal data (image, DSM, DTM) with options for data augmentation 
    techniques such as CutMix, Mixup, and basic image augmentations.

    Attributes:
        img_paths (list): Paths to image files.
        dsm_paths (list): Paths to DSM files.
        dtm_paths (list): Paths to DTM files.
        labels (list): List of labels for classification.
        batch_size (int): Number of samples per batch.
        cutmix_fusion (bool): If True, applies CutMix fusion on images and DEMs.
        augment (bool): If True, applies classic augmentation techniques.
        use_cutmix (bool): If True, applies CutMix augmentation on images and DEMs.
        use_mixup (bool): If True, applies Mixup augmentation on images and DEMs.
        pad (bool): If True, pads the last batch to match the batch size.
        img_size (int): Target image size for resizing.
    """
    def __init__(self, img_paths, dsm_paths, dtm_paths, water_paths, labels, batch_size, cutmix_fusion=False, augment=False, use_cutmix=False, use_mixup=False, pad=False, img_size=256):
        self.img_paths = img_paths
        self.dsm_paths = dsm_paths
        self.dtm_paths = dtm_paths
        self.water_paths = water_paths
        self.labels = labels
        self.batch_size = batch_size
        self.cutmix_fusion = cutmix_fusion
        self.augment = augment
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup
        self.pad = pad
        self.img_size = img_size
        self.indices = np.arange(len(self.img_paths))

    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.ceil(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):        
        """
        Generates one batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            tuple: Tuple containing batch of images, DEMs, and labels.
        """
        # Generate batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.img_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load and preprocess batch data
        batch_images, batch_dsms, batch_dtms, batch_waters, batch_labels = self.__data_generation(batch_indices)

        # Combine DSM and DTM for DEM data
        batch_dems = np.concatenate([batch_dsms, batch_dtms, batch_waters], axis=-1)
        
        # Apply cutmix augmentation if specified
        if self.use_cutmix:
            batch_images, batch_dems, batch_labels = self.__apply_cutmix(batch_images, batch_dems, batch_labels)

        # Apply mixup augmentation if specified
        if self.use_mixup:
            batch_images, batch_dems, batch_labels = self.__apply_mixup(batch_images, batch_dems, batch_labels)

        # Apply augmentations if specified
        if self.augment:
            batch_images, batch_dems = self.__classic_augment_batch(batch_images, batch_dems)

        # Pad if the last batch has fewer samples than batch_size
        if self.pad and (len(batch_images) < self.batch_size):
            pad_size = self.batch_size - len(batch_images)
            batch_images = np.pad(batch_images, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
            batch_dems = np.pad(batch_dems, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
            batch_labels = np.pad(batch_labels, (0, pad_size), mode='constant')

        # Apply CutMix fusion if enabled
        if self.cutmix_fusion:
            batch_images = self.__apply_cutmix_fusion(batch_images, batch_dems)
            return np.array(batch_images), np.array(batch_labels)
        else:
            return (np.array(batch_images), np.array(batch_dems)), np.array(batch_labels)
            

    def on_epoch_end(self):
        """Shuffles indices at the end of each epoch for randomization."""
        np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        """
        Generates data for the given batch indices.

        Args:
            batch_indices (list): List of indices for the current batch.

        Returns:
            tuple: Tuple containing arrays of images, DSMs, DTMs, and labels for the batch.
        """
        batch_images = []
        batch_dsms = []
        batch_dtms = []
        batch_waters = []
        batch_labels = []

        for idx in batch_indices:
            # Load and preprocess images and DEMs
            img = self.load_and_preprocess_image(self.img_paths[idx])
            dsm = self.load_and_preprocess_image(self.dsm_paths[idx], channels=1)
            dtm = self.load_and_preprocess_image(self.dtm_paths[idx], channels=1)
            water = self.load_and_preprocess_image(self.water_paths[idx], channels=1)

            label = 1.0 if self.labels[idx] > 0 else 0.0  # Binary classification based on label

            batch_images.append(img)
            batch_dsms.append(dsm)
            batch_dtms.append(dtm)
            batch_waters.append(water)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_dsms), np.array(batch_dtms), np.array(batch_dtms), np.array(batch_labels)

    def ensure_string_path(self, path):
        if isinstance(path, bytes):
            return path.decode('utf-8')
        return path

    def remove_nan(self, image, nan=-32767.0):
        image = np.nan_to_num(image, nan=-32767.0)
        return image

    def load_and_preprocess_image(self, image_path, channels=10):
        """
        Loads and preprocesses an image or DEM file.

        Args:
            image_path (str): Path to the image file.
            channels (int): Number of channels for the image.

        Returns:
            np.array: Preprocessed image or DEM array.
        """
        image_path = self.ensure_string_path(image_path)  
        
        dataset = gdal.Open(str(image_path))
        if dataset is None:
            raise FileNotFoundError(f"Failed to open image file: {image_path}")

        # Load all bands and resize image to target size
        bands = dataset.RasterCount
        image = np.stack([dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=-1)
        image = image.astype(np.float32)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image = self.remove_nan(image)
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))  # Normalize

        # Adjust to specified channels if necessary
        if channels == 1:
            image = np.expand_dims(image, axis=-1)
        return image

    def __classic_augment_batch(self, batch_images, batch_dsms):
        """
        Apply classic augmentations (flip, rotation, brightness, contrast) to each image in the batch.
    
        Parameters:
            batch_images (list of tf.Tensor): List of images in the batch.
            batch_dsms (list of tf.Tensor): List of DSMs (Digital Surface Models) in the batch.
    
        Returns:
            tuple: Augmented images and DSMs in the same order.
        """
        # Apply augmentations on each image in the batch
        for i in range(len(batch_images)):
            if random.random() > 0.5:
                batch_images[i] = tf.image.flip_left_right(batch_images[i])
                batch_dsms[i] = tf.image.flip_left_right(batch_dsms[i])
            if random.random() > 0.5:
                batch_images[i] = tf.image.flip_up_down(batch_images[i])
                batch_dsms[i] = tf.image.flip_up_down(batch_dsms[i])

            angle = random.uniform(-20, 20)
            batch_images[i] = tfa.image.rotate(batch_images[i], angle)
            batch_dsms[i] = tfa.image.rotate(batch_dsms[i], angle)

            batch_images[i] = tf.image.random_brightness(batch_images[i], 0.1)
            batch_images[i] = tf.image.random_contrast(batch_images[i], 0.9, 1.1)

        return batch_images, batch_dsms

    def __apply_cutmix(self, batch_images, batch_dems, batch_labels, alpha=0.25, beta=0.25):
        """
        Apply CutMix augmentation on the batch of images, DEM data, and labels.
    
        Parameters:
            batch_images (list of tf.Tensor): List of images in the batch.
            batch_dems (list of tf.Tensor): List of DEMs (Digital Elevation Models) in the batch.
            batch_labels (list of tf.Tensor): List of labels corresponding to each image-DEM pair.
            alpha (float): Alpha parameter for the beta distribution.
            beta (float): Beta parameter for the beta distribution.
    
        Returns:
            tuple: Augmented images, DEMs, and labels with CutMix applied.
        """
        random_indices = np.arange(len(batch_images))
        np.random.shuffle(random_indices)
        mixed_images, mixed_dems, mixed_labels = [], [], []

        for i in range(len(batch_images)):
            img1, dem1, label1 = batch_images[i], batch_dems[i], batch_labels[i]
            img2, dem2, label2 = batch_images[random_indices[i]], batch_dems[random_indices[i]], batch_labels[random_indices[i]]

            lambda_value = self.sample_beta_distribution(1, alpha, beta)
            
            # Define Lambda
            lambda_value = lambda_value[0]
            
            x1, y1, height, width = self.get_box(lambda_value, self.img_size)

            imgcrop1 = tf.image.crop_to_bounding_box(img1, y1, x1, height, width)
            img1 = img1 - tf.image.pad_to_bounding_box(
                imgcrop1, y1, x1, self.img_size, self.img_size
            )
            demcrop1 = tf.image.crop_to_bounding_box(dem1, y1, x1, height, width)
            dem1 = dem1 - tf.image.pad_to_bounding_box(
                demcrop1, y1, x1, self.img_size, self.img_size
            )
            
            img_cutmix_crop1 = tf.image.crop_to_bounding_box(img2, y1, x1, height, width)
            img_cutmix = img1 + tf.image.pad_to_bounding_box(
                img_cutmix_crop1, y1, x1, self.img_size, self.img_size
            )
            dem_cutmix_crop1 = tf.image.crop_to_bounding_box(dem2, y1, x1, height, width)
            dem_cutmix = dem1 + tf.image.pad_to_bounding_box(
                dem_cutmix_crop1, y1, x1, self.img_size, self.img_size
            )

            lambda_value = 1 - (height * width) / (self.img_size ** 2)
            mixed_images.append(img_cutmix)
            mixed_dems.append(dem_cutmix)
            mixed_labels.append(lambda_value * label1 + (1 - lambda_value) * label2)

        return np.array(mixed_images), np.array(mixed_dems), np.array(mixed_labels)

    def __apply_mixup(self, batch_images, batch_dems, batch_labels, alpha=0.25, beta=0.25):
        """
        Apply Mixup augmentation on the batch of images, DEM data, and labels.
    
        Parameters:
            batch_images (np.array): Array of image data for the batch.
            batch_dems (np.array): Array of DEM data for the batch.
            batch_labels (np.array): Array of labels for the batch.
            alpha (float): The alpha parameter for the beta distribution in Mixup.
    
        Returns:
            tuple: (mixed_images, mixed_dems, mixed_labels)
        """
        # Generate random indices to pair up images, DEMs, and labels for Mixup
        batch_size = batch_images.shape[0]
        random_indices = np.random.permutation(batch_size)
        
        # Prepare shuffled versions of images, DEMs, and labels for Mixup
        images_shuffled = batch_images[random_indices]
        dems_shuffled = batch_dems[random_indices]
        labels_shuffled = batch_labels[random_indices]
        
        # Sample lambda for Mixup
        l = self.sample_beta_distribution(batch_size, alpha, beta)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))  # Reshape for broadcasting in images
        y_l = tf.reshape(l, (batch_size,))  # Reshape for broadcasting in images
        
        # Apply Mixup to images and DEMs
        mixed_images = batch_images * x_l + images_shuffled * (1 - x_l)
        mixed_dems = batch_dems * x_l + dems_shuffled * (1 - x_l)
        mixed_labels = batch_labels * y_l + labels_shuffled * (1 - y_l)
        
        return np.array(mixed_images), np.array(mixed_dems), np.array(mixed_labels)


    def __apply_cutmix_fusion(self, batch_images, batch_dems, method="patch", alpha=0.25, beta=0.25, n=4):
        """
        Apply CutMix fusion technique on images and DEM data.
    
        Parameters:
            batch_images (list of tf.Tensor): List of images in the batch.
            batch_dems (list of tf.Tensor): List of DEMs (Digital Elevation Models) in the batch.
            method (str): Fusion method ("classic" or "patch").
            alpha (float): Alpha parameter for beta distribution.
            beta (float): Beta parameter for beta distribution.
            n (int): Number of patches for patch-based fusion.
    
        Returns:
            list: List of fused images.
        """
        zip_ = zip(batch_images, batch_dems)

        fused_images = []

        for (img, dem) in zip_:
            fused_img = self.classic_cutmix_fusion(img, dem, alpha=alpha, beta=beta) \
                if method=='classic' \
                else self.patch_cutmix_fusion(img, dem, n=n, alpha=alpha, beta=beta)
            fused_images.append(fused_img)
        return fused_images

    def sample_beta_distribution(self, size, concentration_0=0.2, concentration_1=0.2):
        """
        Sample values from a beta distribution.
    
        Parameters:
            size (int): Number of samples to generate.
            concentration_0 (float): Concentration parameter for beta distribution.
            concentration_1 (float): Concentration parameter for beta distribution.
    
        Returns:
            tf.Tensor: Array of sampled values.
        """
        gamma_1_sample = tf.random.gamma([size], concentration_1)
        gamma_2_sample = tf.random.gamma([size], concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def get_box(self, lambda_value, img_size):
        """
        Calculate coordinates for a bounding box given a lambda value.
    
        Parameters:
            lambda_value (float): Lambda for box scaling.
            img_size (int): Image size (assumes square).
    
        Returns:
            tuple: Coordinates and dimensions of the bounding box (x, y, height, width).
        """
        cut_rat = tf.sqrt(1.0 - lambda_value)
        cut_w = tf.cast(img_size * cut_rat, "int32")
        cut_h = tf.cast(img_size * cut_rat, "int32")
        cut_x = tf.random.uniform((1,), minval=0, maxval=img_size, dtype=tf.int32)[0]
        cut_y = tf.random.uniform((1,), minval=0, maxval=img_size, dtype=tf.int32)[0]
        boundaryx1 = tf.clip_by_value(cut_x - cut_w // 2, 0, img_size)
        boundaryy1 = tf.clip_by_value(cut_y - cut_h // 2, 0, img_size)
        bbx2 = tf.clip_by_value(cut_x + cut_w // 2, 0, img_size)
        bby2 = tf.clip_by_value(cut_y + cut_h // 2, 0, img_size)
        target_h = tf.maximum(bby2 - boundaryy1, 1)
        target_w = tf.maximum(bbx2 - boundaryx1, 1)
        return boundaryx1, boundaryy1, target_h, target_w
    
    def classic_cutmix_fusion(self, image1, image2, alpha=0.25, beta=0.25):
        """
        Perform CutMix fusion on two images with different channel counts.
        
        Parameters:
            image1 (tf.Tensor): The first image tensor (e.g., 10-band).
            image2 (tf.Tensor): The second image tensor (e.g., 3-band).
            fusion_type (str): "reduce" to reduce image1 to match image2, or "expand" to match image2 to image1.
            alpha (float): The alpha parameter for the beta distribution in CutMix.
            beta (float): The beta parameter for the beta distribution in CutMix.
            
        Returns:
            tf.Tensor: The fused CutMix tensor.
        """
        # Ensure images are 4D for batch processing if not already
        if len(image1.shape) == 3:
            image1 = tf.expand_dims(image1, axis=0)
        if len(image2.shape) == 3:
            image2 = tf.expand_dims(image2, axis=0)
    
        # Handle different channel sizes
        image1_reduced = pca_reduce(image1, n_components=image2.shape[-1])
        image1_reduced = tf.cast(image1_reduced, dtype=tf.float32) 
    
        # Sample lambda from a beta distribution for CutMix
        lambda_value = self.sample_beta_distribution(1, alpha, beta)
        # cut_ratio = tf.sqrt(1.0 - lambda_value)
        img_size = image1.shape[1]
    
        # Define cutmix box
        x1, y1, height, width = get_box(lambda_value[0], img_size)
        imgcrop1 = tf.image.crop_to_bounding_box(image1_reduced, y1, x1, height, width)
        image1_reduced = image1_reduced - tf.image.pad_to_bounding_box(
            imgcrop1, y1, x1, img_size, img_size
        )
    
        img_cutmix_crop1 = tf.image.crop_to_bounding_box(image2, y1, x1, height, width)
        fused_image = image1_reduced + tf.image.pad_to_bounding_box(
            img_cutmix_crop1, y1, x1, img_size, img_size
        )
    
        return tf.squeeze(fused_image, axis=0)
    
    def patch_cutmix_fusion(self, image1, image2, n=4, alpha=0.25, beta=0.25):
        """
        Perform grid-based CutMix fusion on two images with different channel counts.
        
        Parameters:
            image1 (tf.Tensor): The first image tensor (e.g., 10-band).
            image2 (tf.Tensor): The second image tensor (e.g., 3-band).
            n (int): Number of divisions along each dimension, creating an n x n grid of patches.
            alpha (float): The alpha parameter for the beta distribution in CutMix.
            beta (float): The beta parameter for the beta distribution in CutMix.
            
        Returns:
            tf.Tensor: The fused CutMix tensor.
        """
        # Ensure images are 4D for batch processing if not already
        if len(image1.shape) == 3:
            image1 = tf.expand_dims(image1, axis=0)
        if len(image2.shape) == 3:
            image2 = tf.expand_dims(image2, axis=0)
        
        # Handle different channel sizes by reducing image1 channels with PCA to match image2
        image1_reduced = self.pca_reduce(image1, n_components=image2.shape[-1])
        image1_reduced = tf.cast(image1_reduced, dtype=tf.float32) 
    
        # Sample the proportion of squares to replace from image1 using a beta distribution
        lambda_value = self.sample_beta_distribution(1, alpha, beta)
        num_squares = int(n * n * tf.clip_by_value(lambda_value[0], 0.25, 0.75))
        
        img_size = image1.shape[1]
        patch_size = tf.cast(img_size // n, tf.int32)
    
        # Generate random square indices to swap between image1 and image2
        all_indices = np.array([(i, j) for i in range(n) for j in range(n)])
        selected_indices = tf.random.shuffle(all_indices)[:num_squares]
    
        # Initialize fused image with image1_reduced
        fused_image = image1_reduced
    
        # Replace selected squares in image1_reduced with corresponding squares from image2
        for idx in selected_indices:
            x, y = idx
            x = tf.cast(x, tf.int32)
            y = tf.cast(y, tf.int32)
            x1 = tf.cast(x * patch_size, tf.int32)
            y1 = tf.cast(y * patch_size, tf.int32)
                  
            imgcrop2 = tf.image.crop_to_bounding_box(image2, y1, x1, patch_size, patch_size)
            fused_image = fused_image - tf.image.pad_to_bounding_box(
                tf.image.crop_to_bounding_box(fused_image, y1, x1, patch_size, patch_size), y1, x1, img_size, img_size
            ) + tf.image.pad_to_bounding_box(imgcrop2, y1, x1, img_size, img_size)
    
        return tf.squeeze(fused_image, axis=0)
        
    
    def pca_reduce(self, image, n_components=3):
        """
        Reduce image channels using PCA.
        
        Parameters:
            image (tf.Tensor): The input image tensor (batch_size, height, width, channels).
            n_components (int): The target number of components (channels) after reduction.
            
        Returns:
            tf.Tensor: The image reduced to `n_components` channels.
        """
        batch_size, height, width, channels = image.shape
        reshaped_image = tf.reshape(image, (batch_size, -1, channels))
        reduced_images = []
        
        for i in range(batch_size):
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(reshaped_image[i])
            reduced_images.append(tf.reshape(reduced, (height, width, n_components)))
        
        return tf.stack(reduced_images, axis=0)


class CombinedDataGenerator(tf.keras.utils.Sequence):
    """
    A Keras Sequence generator that combines data from two generators, allowing
    for optional fusion using CutMix and shuffling of data at the batch level.

    Parameters:
    - gen1: Generator, the first data generator instance to use.
    - gen2: Generator, the second data generator instance to use.
    - shuffle: bool, optional (default=True), whether to shuffle the data after each epoch.
    - cutmix_fusion: bool, optional (default=False), whether to concatenate images only or to include other data (e.g., DEM data).

    Methods:
    - __len__: Returns the number of batches per epoch, based on the smaller generator's length.
    - __getitem__: Retrieves and optionally shuffles a batch of data from both generators.
    - on_epoch_end: Shuffles data in each generator at the end of each epoch.
    """
    def __init__(self, gen1, gen2, shuffle=True, cutmix_fusion=False):
        self.gen1 = gen1
        self.gen2 = gen2
        self.shuffle = shuffle
        self.cutmix_fusion = cutmix_fusion
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch, which is based on the smaller
        of the two generators to ensure consistent batch size.
        """
        # The length should match the smaller of the two generators to avoid an incomplete batch
        return min(len(self.gen1), len(self.gen2))

    def __getitem__(self, index):
        """
        Retrieves a batch of data from each generator, concatenates the results,
        and shuffles them if specified.

        Parameters:
        - index: int, the index of the batch to retrieve.

        Returns:
        - Tuple of concatenated (and optionally shuffled) images and labels.
          If cutmix_fusion is False, returns ((images, dem), labels).
          Otherwise, returns (images, labels).
        """
        # Retrieve batches from both generators
        batch1 = self.gen1[index]
        batch2 = self.gen2[index]

        # Concatenate data and labels from both generators
        if self.cutmix_fusion:
            images = np.concatenate([batch1[0], batch2[0]], axis=0)
        else:
            images = np.concatenate([batch1[0][0], batch2[0][0]], axis=0)
            dem = np.concatenate([batch1[0][1], batch2[0][1]], axis=0)
        labels = np.concatenate([batch1[1], batch2[1]], axis=0)

        # Shuffle data if required
        if self.shuffle:
            indices = np.arange(images.shape[0])
            np.random.shuffle(indices)
            images = images[indices]
            if not self.cutmix_fusion:
                dem = dem[indices]
            labels = labels[indices]

        if self.cutmix_fusion: 
            return images, labels
        else:
            return (images, dem), labels

    def on_epoch_end(self):
        """
        Shuffles data at the end of each epoch if each generator has an `on_epoch_end` method.
        """
        # Shuffle both generators on epoch end if they support shuffling
        if hasattr(self.gen1, 'on_epoch_end'):
            self.gen1.on_epoch_end()
        if hasattr(self.gen2, 'on_epoch_end'):
            self.gen2.on_epoch_end()



def get_combined_generator(img_paths, dsm_paths, dtm_paths, water_paths, labels, batch_size, cutmix_fusion=False, augment_first_gen=False, augment_second_gen=True, mixup_first_gen=False, mixup_second_gen=False, cutmix_second_gen=False, img_size=256):
    """
    Creates a combined data generator by initializing two multimodal data generators,
    one with augmentation and one without, and combining them using CombinedDataGenerator.

    Parameters:
    - img_paths: list, paths to image files.
    - dsm_paths: list, paths to DSM files.
    - dtm_paths: list, paths to DTM files.
    - labels: list, the labels associated with each data sample.
    - batch_size: int, the size of each batch.
    - cutmix_fusion: bool, optional (default=False), whether to use CutMix fusion.
    - augment_first_gen: bool, optional (default=False), whether to augment the first generator.
    - augment_second_gen: bool, optional (default=True), whether to augment the second generator.
    - mixup_first_gen: bool, optional (default=False), whether to apply MixUp in the first generator.
    - mixup_second_gen: bool, optional (default=False), whether to apply MixUp in the second generator.
    - cutmix_second_gen: bool, optional (default=False), whether to apply CutMix in the second generator.
    - img_size: int, optional (default=256), the size to resize images.

    Returns:
    - CombinedDataGenerator, a generator that combines the two initialized generators.
    """
    train_gen_non_augmented = MultimodalDataGenerator(
        img_paths=img_paths,
        dsm_paths=dsm_paths,
        dtm_paths=dtm_paths,
        water_paths=water_paths,
        labels=labels,
        batch_size=batch_size // 2,
        augment=augment_first_gen,
        use_mixup=mixup_first_gen,
        use_cutmix=False,
        cutmix_fusion=cutmix_fusion,
        img_size=img_size
    )

    augmented_shuffling_indices_A = random.sample(range(len(img_paths)), 
                                                len(img_paths))
    image_paths_B = [img_paths[i] for i in augmented_shuffling_indices_A]
    dsm_paths_B = [dsm_paths[i] for i in augmented_shuffling_indices_A]
    dtm_paths_B = [dtm_paths[i] for i in augmented_shuffling_indices_A]
    water_paths_B = [water_paths[i] for i in augmented_shuffling_indices_A]
    labels_B = [labels[i] for i in augmented_shuffling_indices_A]
    
    train_gen_augmented = MultimodalDataGenerator(
        img_paths=image_paths_B,
        dsm_paths=dsm_paths_B,
        dtm_paths=dtm_paths_B,
        water_paths=water_paths_B,
        labels=labels_B,
        batch_size=batch_size // 2,
        augment=augment_second_gen,
        use_mixup=mixup_second_gen,
        use_cutmix=cutmix_second_gen,
        cutmix_fusion=cutmix_fusion,
        img_size=img_size
    )

    # Create the combined generator
    train_gen_combined = CombinedDataGenerator(train_gen_augmented, train_gen_non_augmented, cutmix_fusion=cutmix_fusion)

    return train_gen_combined

# def ensure_string_path(path):
#     if isinstance(path, bytes):
#         return path.decode('utf-8')
#     return path

# def augment(image, dem):
#     # Randomly flip the image and mask horizontally
#     if random.random() > 0.5:
#         image = tf.image.flip_left_right(image)
#         dem = tf.image.flip_left_right(dem)

#     # Randomly flip the image and mask vertically
#     if random.random() > 0.5:
#         image = tf.image.flip_up_down(image)
#         dem = tf.image.flip_up_down(dem)

#     # Random rotation between -20 and +20 degrees
#     angle = random.uniform(-20, 20)
#     image = tfa.image.rotate(image, angle)
#     dem = tfa.image.rotate(dem, angle)

#     # Randomly adjust brightness
#     image = tf.image.random_brightness(image, 0.1)

#     # Randomly adjust contrast
#     image = tf.image.random_contrast(image, 0.9, 1.1)

#     # Randomly adjust saturation (if the image has color channels)
#     # if image.shape[-1] > 1:
#     #     image = tf.image.random_saturation(image, 0.9, 1.1)
    
#     return image, dem

# def load_and_preprocess_image(image_path, im_size=256):
#     # Convert TensorFlow tensor to a string
#     # image_path = image_path.numpy().decode('utf-8')
#     image_path = ensure_string_path(image_path)  
    
#     # Open the image file with GDAL
#     dataset = gdal.Open(str(image_path))
#     if dataset is None:
#         raise FileNotFoundError(f"Failed to open image file: {image_path}")

#     # Read the image bands
#     bands = dataset.RasterCount
#     image = np.stack([dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=-1)
    
#     # Convert to float32 and normalize
#     image = image.astype(np.float32)
#     # print('before', image.shape)
#     image = cv2.resize(image, dsize=(im_size, im_size), interpolation=cv2.INTER_LINEAR)
#     # print('after', image.shape)
   
#     return image

# def remove_nan(image, nan=-32767.0):
#     image = np.nan_to_num(image, nan=-32767.0)
#     return image

# @tf.autograph.experimental.do_not_convert
# def preprocess(image_path, dsm_path, dtm_path, water_path, label, augment_data=False):
#     def load_image(image_path):
#         return tf.numpy_function(load_and_preprocess_image, [image_path], tf.float32)

#     def tf_remove_nan(image):
#         return tf.numpy_function(remove_nan, [image], tf.float32)
        
#     # Load image
#     image = load_image(image_path)
#     image = tf_remove_nan(image)
#     image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
#     # print('image', image.shape)
#     image.set_shape([None, None, 10])
#     # print('image', image.shape)

#     dsm = load_image(dsm_path)
#     dsm = tf_remove_nan(dsm)
#     dsm = (dsm - tf.reduce_min(dsm)) / (tf.reduce_max(dsm) - tf.reduce_min(dsm))
#     # print('dsm', dsm.shape)
#     dsm = tf.expand_dims(dsm, axis=-1)
#     dsm.set_shape([None, None, 1])
#     # print('dsm', dsm.shape)

    
#     dtm = load_image(dtm_path)
#     dtm = tf_remove_nan(dtm)
#     dtm = (dtm - tf.reduce_min(dtm)) / (tf.reduce_max(dtm) - tf.reduce_min(dtm))
#     # print('dtm', dtm.shape)
#     dtm = tf.expand_dims(dtm, axis=-1)
#     dtm.set_shape([None, None, 1])
#     # print('dtm', dtm.shape)

#     water = load_image(water_path)
#     water = tf_remove_nan(water)
#     water = (water - tf.reduce_min(water)) / (tf.reduce_max(water) - tf.reduce_min(water))
#     # print('dtm', dtm.shape)
#     water = tf.expand_dims(water, axis=-1)
#     water.set_shape([None, None, 1])
#     # print('dtm', dtm.shape)


#     dem = tf.concat([dsm, dtm, water], axis=-1)

#     label = tf.cast(label > 0, tf.int64)
#     label.set_shape([])

#     # Apply augmentation if enabled
#     if augment_data:
#         image, dem = augment(image, dem)
    
#     return (image, dem), label
