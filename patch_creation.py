import rasterio
import pandas as pd
import geopandas as gpd
from rasterio.transform import rowcol, from_origin
import numpy as np
from rasterio.windows import Window
import os
import matplotlib.pyplot as plt
import glob
import random
from tqdm import tqdm
from utils import overlay_masks, _save_patch, read_image
from remote_sensing_utils import resample_dem_to_match, clip_vectors_to_image, vector_to_raster

class CreatePatches:
    def __init__(self, image_path, dem_path, points_path, labels_path, output_dir, patch_size, site, campaign):
        """
        Initialize the CreatePatch class with paths and parameters for creating patches.
        
        Parameters:
        - image_path: Path to the multispectral image.
        - dsm_path: Path to the Digital Surface Model (DSM) image.
        - points_path: Path to the vector data (geopackage file).
        - output_dir: Directory where patches will be saved.
        - patch_size: Size of each patch (e.g., 256x256).

        Example of use:
        original_image_path = '../donnees_terrain/donnees_brutes_initiales/Strange_Ortho_MS_mission1/2024-04-29_Timbertiere_MS_DualMX_ortho_8cm.tif'
        dsm_path = '../donnees_terrain/donnees_brutes_initiales/20240626_Timbertiere_LiDAR/2024-06-26_Timbertiere_dsm.tif'
        points_path = '../donnees_terrain/donnees_brutes_initiales/Timbertiere04.24.gpkg'
        output_dir = '../code/output_patches_bis/'
        patch_size = 512
        
        create_patches = CreatePatches(original_image_path, dsm_path, points_path, output_dir, patch_size)
        patches, dem_patches = create_patches.create_patches()
        create_patches.show_patches(output_dir + "image_patches/", num_patches=8, bands_to_show=[5, 3, 1])
        """
        self.image_path = image_path
        self.dem_path = dem_path
        self.points_path = points_path
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.site = site
        self.campaign = campaign

    @staticmethod
    def show_patches(patch_dir, num_patches=5, max_images_per_row=4, bands_to_show=[1]):
        """
        Show a specified number of patches from the given directory, with a maximum of 4 images per row.
        Handles either 1-band grayscale or 3-band RGB images based on the bands_to_show parameter.
        
        Parameters:
        - patch_dir: Directory where patches are saved.
        - num_patches: Number of patches to display.
        - max_images_per_row: Maximum number of images per row.
        - bands_to_show: List of band indices to use for displaying (1-based indices).
                          Provide either 1 band for grayscale or 3 bands for RGB images.
        """
        # Validate band indices
        if len(bands_to_show) not in [1, 3]:
            raise ValueError("Please provide exactly 1 or 3 bands for display.")
        
        # Get a sorted list of patch files from the directory
        patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.tif')))
        
        # Select only the number of patches to display
        selected_files = patch_files[:num_patches]
        
        # Calculate number of rows needed
        num_rows = (len(selected_files) + max_images_per_row - 1) // max_images_per_row
        
        plt.figure(figsize=(15, num_rows * 5))  # Adjust figure size based on number of rows
        
        for idx, patch_file in enumerate(selected_files):
            with rasterio.open(patch_file) as patch:
                # Read the specified bands
                band_data = [patch.read(band) for band in bands_to_show]
                
                if len(bands_to_show) == 1:
                    # Handle single-band (grayscale) images
                    single_band_image = band_data[0]
                    
                    plt.subplot(num_rows, max_images_per_row, idx + 1)
                    plt.imshow(single_band_image, cmap='gray', vmin=0, vmax=single_band_image.max())
                    plt.title(f'Patch {idx + 1}')
                    plt.axis('off')
                
                elif len(bands_to_show) == 3:
                    # Handle three-band (RGB) images
                    rgb_image = np.stack(band_data, axis=-1)
                    
                    # Normalize RGB image for display
                    rgb_image = np.clip(rgb_image, 0, 255)
                    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255
                    rgb_image = rgb_image.astype(np.uint8)
                    
                    plt.subplot(num_rows, max_images_per_row, idx + 1)
                    plt.imshow(rgb_image)
                    plt.title(f'Patch {idx + 1}')
                    plt.axis('off')
        
        plt.tight_layout()
        plt.show()


    def create_patches(self, create_labels, dem_basename=None, dem_folder_name=None, resample=True, save_patches=True, save_dem_patches=True):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir + dem_folder_name, exist_ok=True)
        os.makedirs(self.output_dir + "image_patches", exist_ok=True)
        
        # Load the multispectral image
        with rasterio.open(self.image_path) as img_src:
            img_crs = img_src.crs  # CRS (Coordinate Reference System) of the image
            img_transform = img_src.transform  # Affine transform for converting coordinates to pixel indices
            img_bands = img_src.count  # Number of bands in the image (e.g., RGB or multispectral bands)
            img_bounds = img_src.bounds
            img_meta = img_src.meta
            img_profile =img_src.profile
            img = img_src.read()
    
        with rasterio.open(self.dem_path) as dem_src:
            dem_transform = dem_src.transform
            dem_profile =dem_src.profile
            dem = dem_src.read()
        
        # Load the points from the GeoPackage file
        points_gdf = gpd.read_file(self.points_path)
        
        # Ensure the points are in the same CRS as the image
        if points_gdf.crs != img_crs:
            points_gdf = points_gdf.to_crs(img_crs)
    
        points_gdf = clip_vectors_to_image(points_gdf, img_bounds)
    
        if resample:
            dem = resample_dem_to_match(
                src_path=self.dem_path,
                dest_transform=img_transform,
                dest_crs=img_crs,
                dest_width=img_meta['width'],
                dest_height=img_meta['height']
            )

        if create_labels:
            if not os.path.exists(self.labels_path):
                labels = pd.DataFrame(columns=["id", "site", "campaign", "Ephemeropteres", "Plecopeteres", "Trichopteres"])
            else:
                labels = pd.read_csv(self.labels_path, index_col="Unnamed: 0")
    
        patches = []
        dem_patches = []

        for idx, point in tqdm(points_gdf.iterrows()):
            x, y = point.geometry.x, point.geometry.y  # Get the x and y coordinates of the point
            if create_labels:
                E = point.Nb_Ephemeropteres
                P = point.Nb_Plecopteres
                T = point.Nb_Trichopteres
            
            # Convert geographic coordinates to pixel indices in the image
            px, py = rowcol(img_transform, x, y)
            # Extract the patches from the images
            # patch = img[:, px - self.patch_size//2: px + self.patch_size//2, py - self.patch_size//2: py + self.patch_size//2]
            # dem_patch = dem[px - self.patch_size//2: px + self.patch_size//2, py - self.patch_size//2: py + self.patch_size//2]
            patch = img[:, np.max([0, px - self.patch_size//2]): px + self.patch_size//2, np.max([0, py - self.patch_size//2]): py + self.patch_size//2]
            dem_patch = dem[np.max([0, px - self.patch_size//2]): px + self.patch_size//2,  np.max([0, py - self.patch_size//2]): py + self.patch_size//2]
            
    
            patches.append(patch)
            dem_patches.append(dem_patch)
            
            if create_labels:
                labels.loc[labels.shape[0]+1, :] = [idx+1, self.site, self.campaign, E, P, T]
                labels.to_csv(self.labels_path)
    
            if save_patches:
                _save_patch(patch=patch, 
                            profile=img_profile, 
                            index=idx+1,
                            base_name=self.campaign + "_" + self.site + "_" + "patch", 
                            output_dir=self.output_dir + "image_patches",
                           )
    
            if save_dem_patches:
                _save_patch(patch=dem_patch, 
                            profile=dem_profile, 
                            index=idx+1,
                            base_name=self.campaign + "_" + self.site + "_" + dem_basename, 
                            output_dir=self.output_dir + dem_folder_name,
                            multispectrale=False
                           )

        if create_labels:
            return patches, dem_patches, labels
        else:
            return patches, dem_patches


class CreateWaterPatches:
    def __init__(self, ms_image_path, dem_image_path, water_vectors_path, patch_size, patch_output_dir, 
                 resample_output_dir=None, resample_output_file=None, site_name=None):
        """
        Initialize the class with paths and parameters.
        
        Parameters:
        - ms_image_path: Path to the multispectral image.
        - dem_image_path: Path to the DEM image.
        - water_vectors_path: Path to the GPKG file containing water vectors.
        - patch_size: Tuple specifying the size of each patch.
        - patch_output_dir: Directory to store the output patches.
        - resample_output_dir: Directory to store the resampled DEM (if needed).
        - resample_output_file: File name for the resampled DEM.
        - site_name: Name of the site being processed.
        """
        self.ms_image_path = ms_image_path
        self.dem_image_path = dem_image_path
        self.water_vectors_path = water_vectors_path
        self.patch_size = patch_size
        self.patch_output_dir = patch_output_dir
        self.resample_output_dir = resample_output_dir
        self.resample_output_file = resample_output_file
        self.site_name = site_name

    @staticmethod
    def get_metadata(path):
        """
        Get metadata from a raster image using rasterio.
        
        Parameters:
        - path: str, Path to the raster image file.
        
        Returns:
        - profile: dict, Metadata profile of the image (e.g., data type, dimensions).
        - crs: rasterio.CRS, Coordinate Reference System of the image.
        - transform: Affine, Geospatial transformation matrix.
        - width: int, Width of the image (number of columns).
        - height: int, Height of the image (number of rows).
        - res: tuple, Pixel size in (x, y) directions.
        - bounds: Bounding box of the image.
        """
        with rasterio.open(path) as src:
            profile = src.profile
            crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            res = src.res
            bounds = src.bounds
        return profile, crs, transform, width, height, res, bounds

    
    @staticmethod
    def read_vectors(gpkg_path):
        """
        Read vector layers (water features) from a GPKG file using GeoPandas.
        
        Parameters:
        - gpkg_path: str, Path to the GPKG file containing vector data.
        
        Returns:
        - gdf_line: GeoDataFrame, Buffer of the linear water features (e.g., rivers).
        - gdf_poly: GeoDataFrame, Polygonal water bodies (e.g., lakes).
        """
        gdf_line = gpd.read_file(gpkg_path, layer='reseauhydro_lineaire').buffer(3)
        gdf_poly = gpd.read_file(gpkg_path, layer='Surface en eau')
        return gdf_line, gdf_poly




    def extract_patches(self, image, mask, dest_profile=None, multispectrale=True, save_patches=False, base_name='image_patch', save_mask_patches=False):
        """
        Extract non-overlapping patches from an image and the corresponding mask.
        
        Parameters:
        - image: np.ndarray, Image array from which patches are extracted.
        - mask: np.ndarray, Mask array for the image (e.g., water mask).
        - dest_profile: dict, Metadata for the destination patches.
        - multispectrale: bool, If True, treat the image as multispectral (multiple bands).
        - save_patches: bool, If True, save the extracted patches.
        - base_name: str, Base name for the saved patches.
        - save_mask_patches: bool, If True, save the corresponding mask patches.
        
        Returns:
        - patches: list, List of extracted image patches.
        - mask_patches: list, List of extracted mask patches.
        """
        patches = []
        mask_patches = []
        image_shape = image.shape
        image_height = image_shape[0]
        image_width = image_shape[1]
        dest_crs = dest_profile['crs']
        dest_transform = dest_profile['transform']
        n = 0
        
        for i in tqdm(range(0, image_height, self.patch_size)):
            for j in range(0, image_width, self.patch_size):
                patch = image[i:i + self.patch_size, j:j + self.patch_size, :] if len(image_shape) > 2 else image[i:i + self.patch_size, j:j + self.patch_size]
                mask_patch = mask[i:i + self.patch_size, j:j + self.patch_size]

                if save_patches:
                    _save_patch(patch, dest_profile, n, base_name, self.patch_output_dir, multispectrale=multispectrale)
                if save_mask_patches:
                    _save_patch(mask_patch, dest_profile, n, "mask_patch", self.patch_output_dir, multispectrale=False)

                patches.append(patch)
                mask_patches.append(mask_patch)
                n += 1
                
        return patches, mask_patches

    def create_patches(self, create_MS_patches=True, create_dem_patches=True, save_patches=True, image_basename='image_patch', save_dem_patches=True, dem_basename='dem_patch', save_mask_patches=True, resample=True, save_resampled=False, resample_output_dir=None, resample_output_file=None, same_crs_dem_ms=True):
        """
        Main method to generate non-overlapping patches from the multispectral image, DEM, and water mask.
        
        This method extracts patches from the input multispectral (MS) image, the resampled DEM (Digital Elevation Model),
        and corresponding water mask created from vector data (e.g., water bodies and rivers). Patches can be saved as 
        image files, with optional saving for the resampled DEM and mask patches. The function also supports resampling 
        the DEM to match the multispectral image resolution if required.
    
        Parameters:
        - create_MS_patches: bool, If True, extract patches from the multispectral image, default=True.
        - create_dem_patches: bool, If True, extract patches from the resampled DEM, default=True.
        - save_patches: bool, If True, save the extracted multispectral image patches as files, default=True.
        - image_basename: str, Base name for the saved multispectral image patch files, default='image_patch'.
        - save_dem_patches: bool, If True, save the extracted DEM patches as files, default=True.
        - dem_basename: str, Base name for the saved DEM patch files, default='dem_patch'.
        - save_mask_patches: bool, If True, save the corresponding water mask patches for each multispectral image patch, default=True
        - resample: bool, If True, resample the DEM to match the resolution and extent of the multispectral image, default=True
        - save_resampled: bool, If True, save the resampled DEM to a file, default=False
        - same_crs_dem_ms: bool, If True, ensure that the DEM and multispectral image share the same CRS (Coordinate Reference System), default=True
        
        Returns:
        - If both create_MS_patches and create_dem_patches are True:
            - multispectral_patches: list of np.ndarray, List of extracted patches from the multispectral image.
            - dsm_patches: list of np.ndarray, List of extracted patches from the resampled DEM.
            - mask_patches: list of np.ndarray, List of water mask patches corresponding to the multispectral image patches.
        - If only create_MS_patches is True:
            - multispectral_patches: list of np.ndarray, List of extracted patches from the multispectral image.
            - mask_patches: list of np.ndarray, List of water mask patches corresponding to the multispectral image patches.
        - If only create_dem_patches is True:
            - dsm_patches: list of np.ndarray, List of extracted patches from the resampled DEM.
            - mask_patches: list of np.ndarray, List of water mask patches corresponding to the multispectral image patches.
        - If neither create_MS_patches nor create_dem_patches are True:
            - None
        
        Notes:
        - This method assumes that the water vectors are provided as layers in a GPKG file. These vectors are clipped to 
          the bounds of the multispectral image before being rasterized into a mask.
        - The DEM is resampled to match the spatial resolution and extent of the multispectral image if 'resample' is set 
          to True.
        - If 'save_patches' or 'save_dem_patches' are True, the patches will be saved in the specified directory with the 
          given base names.
        """
        os.makedirs(self.patch_output_dir, exist_ok=True)

        MS_profile, MS_crs, MS_transform, MS_width, MS_height, MS_res, MS_bounds = self.get_metadata(self.ms_image_path)

        if resample:
            resampled_dsm = resample_dem_to_match(
                src_path=self.dem_image_path, dest_transform=MS_transform, dest_crs=MS_profile['crs'], 
                dest_width=MS_width, dest_height=MS_height, same_crs=same_crs_dem_ms, save_resampled=save_resampled,
                resample_output_dir=resample_output_dir, resample_output_file=resample_output_file)
            

        gdf_line, gdf_poly = self.read_vectors(self.water_vectors_path)
        gdf_line = clip_vectors_to_image(gdf_line, MS_bounds)
        gdf_poly = clip_vectors_to_image(gdf_poly, MS_bounds)

        water_mask = vector_to_raster([gdf_line, gdf_poly], MS_transform, MS_width, MS_height)

        MS_img = np.transpose(read_image(self.ms_image_path).read(), (1, 2, 0))

        if create_MS_patches:
            multispectral_patches, mask_patches = self.extract_patches(
                MS_img, water_mask, dest_profile=MS_profile, save_patches=save_patches, base_name=image_basename, save_mask_patches=save_mask_patches
            )

        if create_dem_patches:
            dsm_patches, mask_patches = self.extract_patches(
                resampled_dsm, water_mask, dest_profile=MS_profile, multispectrale=False, save_patches=save_dem_patches, base_name=dem_basename, save_mask_patches=False
            )

        if create_MS_patches and create_dem_patches:
            return multispectral_patches, dsm_patches, mask_patches
        elif create_MS_patches and not create_dem_patches:
            return multispectral_patches, mask_patches
        elif create_dem_patches and not create_MS_patches:
            return dsm_patches, mask_patches
        else:
            return None


class CreatePatchesForPseudoLabels:
    def __init__(self, ms_image_path, dem_image_path, patch_size, patch_output_dir, random_points=True):
        """
        Initialize the class with paths and parameters.

        Parameters:
        - ms_image_path: Path to the multispectral image.
        - dem_image_path: Path to the DEM image.
        - patch_size: Tuple specifying the size of each patch.
        - patch_output_dir: Directory to store the output patches.
        """
        self.ms_image_path = ms_image_path
        self.dem_image_path = dem_image_path
        self.patch_size = patch_size
        self.patch_output_dir = patch_output_dir

    @staticmethod
    def get_metadata(path):
        """
        Get metadata from a raster image using rasterio.
        
        Parameters:
        - path: str, Path to the raster image file.
        
        Returns:
        - profile: dict, Metadata profile of the image (e.g., data type, dimensions).
        - crs: rasterio.CRS, Coordinate Reference System of the image.
        - transform: Affine, Geospatial transformation matrix.
        - width: int, Width of the image (number of columns).
        - height: int, Height of the image (number of rows).
        - res: tuple, Pixel size in (x, y) directions.
        - bounds: Bounding box of the image.
        """
        with rasterio.open(path) as src:
            profile = src.profile
            crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            res = src.res
            bounds = src.bounds
        return profile, crs, transform, width, height, res, bounds

    def extract_patches(self, image, nb_random_points=5000, dest_profile=None, base_name='image_patch', save_patches=False, output_dir=None, multispectrale=True):
        """
        Extract overlapping patches from an image.

        Parameters:
        - image: np.ndarray, Image array from which patches are extracted.
        - nb_random_points: int, number of random vignettes to extract. If None, vignettes will be extracted at every pixel.
        - dest_profile: dict, Metadata for the destination patches.
        - base_name: str, Base name for the saved patches.
        - save_patches: bool, If True, save the extracted patches.
        - output_dir: str, Directory to store the output patches.
        - multispectrale: bool, If True, treat the image as multispectral (multiple bands).

        Returns:
        - patches: list, List of extracted image patches.
        """
        if save_patches:
            os.makedirs(output_dir, exist_ok=True)
        
        patches = []
        image_shape = image.shape
        image_height = image_shape[0]
        image_width = image_shape[1]
        n = 0

        if nb_random_points: 
            range_i = list(range(image_height))
            range_j = list(range(image_width))
            random.seed(10)
            random_is = [random.choice(range_i) for i in range(nb_random_points)]
            random_js = [random.choice(range_j) for i in range(nb_random_points)]
            for i, j in tqdm(zip(random_is, random_js)):
                # print(i, j)
                patch = image[np.max([0, i - self.patch_size//2]):np.min([image_height, i + self.patch_size//2]), 
                            np.max([0, j - self.patch_size//2]):np.min([image_width, j + self.patch_size//2]), :] \
                        if len(image_shape) > 2 \
                        else image[np.max([0, i - self.patch_size//2]):np.min([image_height, i + self.patch_size//2]), 
                            np.max([0, j - self.patch_size//2]):np.min([image_width, j + self.patch_size//2])]

                if save_patches:
                    _save_patch(patch, dest_profile, f'{i}_{j}', base_name, output_dir, multispectrale=multispectrale)

                patches.append(patch)
                n += 1
        else:
            for i in tqdm(range(0, image_height)):
                for j in range(0, image_width):
                    patch = image[np.max([0, i - self.patch_size//2]):np.min([image_height, i + self.patch_size//2]), 
                                np.max([0, j - self.patch_size//2]):np.min([image_width, j + self.patch_size//2]), :] \
                            if len(image_shape) > 2 \
                            else image[np.max([0, i - self.patch_size//2]):np.min([image_height, i + self.patch_size//2]), 
                                np.max([0, j - self.patch_size//2]):np.min([image_width, j + self.patch_size//2])]
    
                    if save_patches:
                        _save_patch(patch, dest_profile, f'{i}_{j}', base_name, output_dir, multispectrale=multispectrale)
    
                    patches.append(patch)
                    n += 1
        return patches

    def create_patches(self, create_MS_patches=True, 
                       create_dem_patches=True, 
                       nb_random_points=5000,
                       save_patches=True, 
                       image_basename='image_patch', 
                       dem_folder_name=None,
                       dem_basename='dem_patch', 
                       resample=True, 
                       save_resampled=False, 
                       resample_output_dir=None,
                       resample_output_file=None, 
                       same_crs_dem_ms=True):
        """
        Main method to generate overlapping patches from the multispectral image and DEM.

        This method extracts patches from the input multispectral (MS) image and the DEM.

        Parameters:
        - create_MS_patches: bool, If True, extract patches from the multispectral image, default=True.
        - create_dem_patches: bool, If True, extract patches from the DEM, default=True.
        - nb_random_points: int, number of random vignettes to extract. If None, vignettes will be extracted at every pixel.
        - save_patches: bool, If True, save the extracted patches, default=True.
        - image_basename: str, Base name for the saved multispectral image patch files, default='image_patch'.
        - dem_folder_name: str, Name of the folder  where the dem patches will be stored, default=None,
        - dem_basename: str, Base name for the saved DEM patch files, default='dem_patch'.
        - resample: bool, If True, resample the DEM to match the resolution and extent of the multispectral image, default=True
        - save_resampled: bool, If True, save the resampled DEM to a file, default=False
        - resample_output_dir: Directory to store the resampled DEM (if needed).
        - resample_output_file: File name for the resampled DEM.
        - same_crs_dem_ms: bool, If True, ensure that the DEM and multispectral image share the same CRS (Coordinate Reference System), default=True

        Returns:
        - If both create_MS_patches and create_dem_patches are True:
            - multispectral_patches: list of np.ndarray, List of extracted patches from the multispectral image.
            - dem_patches: list of np.ndarray, List of extracted patches from the DEM.
        - If only create_MS_patches is True:
            - multispectral_patches: list of np.ndarray, List of extracted patches from the multispectral image.
        - If only create_dem_patches is True:
            - dem_patches: list of np.ndarray, List of extracted patches from the DEM.
        """
        os.makedirs(self.patch_output_dir, exist_ok=True)

        # Get metadata and image for multispectral image
        MS_profile, MS_crs, MS_transform, MS_width, MS_height, MS_res, MS_bounds = self.get_metadata(self.ms_image_path)
        MS_img = np.transpose(read_image(self.ms_image_path).read(), (1, 2, 0))

        if resample:
            resampled_dsm = resample_dem_to_match(
                src_path=self.dem_image_path, dest_transform=MS_transform, dest_crs=MS_profile['crs'], 
                dest_width=MS_width, dest_height=MS_height, same_crs=same_crs_dem_ms, save_resampled=save_resampled,
                resample_output_dir=resample_output_dir, resample_output_file=resample_output_file)

        # Extract patches
        multispectral_patches, dem_patches = None, None

        if create_MS_patches:
            multispectral_patches = self.extract_patches(
                MS_img, 
                nb_random_points=nb_random_points,
                dest_profile=MS_profile, 
                base_name=image_basename, 
                save_patches=save_patches, 
                output_dir=self.patch_output_dir + "pseudo_labels_image_patches/"
            )

        if create_dem_patches:
            dem_patches = self.extract_patches(
                resampled_dsm, 
                nb_random_points=nb_random_points,
                dest_profile=MS_profile, 
                base_name=dem_basename, 
                save_patches=save_patches, 
                output_dir=self.patch_output_dir + dem_folder_name, 
                multispectrale=False
            )

        if create_MS_patches and create_dem_patches:
            return multispectral_patches, dem_patches
        elif create_MS_patches:
            return multispectral_patches
        elif create_dem_patches:
            return dem_patches
        else:
            return None


def study_field_train_test_split(image_path, 
                                 points_path, 
                                 train_val_test=True, 
                                 test_split=0.2,
                                 val_split=0.2,
                                 image_basename=None, 
                                 image_output_dir=None, 
                                 points_basename=None, 
                                 points_output_dir=None, 
                                 save_splits=True, 
                                 save_gdf_modified=True):
    """
    Splits an image and its corresponding shapefile into training and testing parts based on spatial location.

    Args:
        image_path (str): Path to the input image (TIFF) file.
        points_path (str): Path to the input shapefile containing points.
        test_split (float, optional): Fraction of points to assign to the test split. Defaults to 0.2.
        val_split (float, optional): Fraction of remaining points to assign to the validation split. Defaults to 0.2.
        train_val_test (bool, optional): If True, perform three-way split. If False, only split into train and test. Defaults to True.
        image_basename (str): Basename for saving output image files.
        image_output_dir (str): Directory to save split image files.
        points_basename (str): Basename for saving modified shapefile.
        points_output_dir (str): Directory to save modified shapefile.
        save_splits (bool, optional): If True, save split image files. Defaults to True.
        save_gdf_modified (bool, optional): If True, save modified shapefile. Defaults to True 

    Returns:
        tuple: (numpy.ndarray, numpy.ndarray, GeoDataFrame) - Arrays of the west (test) and east (train) parts of the image and the modified shapefile with "partition" column added.
    """
    # Step 1: Load the shapefile
    gdf = gpd.read_file(points_path)

    # Step 2: Sort by longitude and find split points
    gdf['x'] = gdf.geometry.x
    gdf_sorted = gdf.sort_values(by="x")

    num_test_points = int(np.ceil(gdf.shape[0] * test_split))
    num_val_points = int(np.ceil((gdf.shape[0] - num_test_points) * val_split)) if train_val_test else 0

    test_points = gdf_sorted[:num_test_points]
    val_points = gdf_sorted[num_test_points:num_test_points + num_val_points] if train_val_test else None
    train_points = gdf_sorted[num_test_points + num_val_points:] if train_val_test else gdf_sorted[num_test_points:]

    # Step 3: Calculate division line
    easternmost_test_x = test_points.geometry.x.max()
    westernmost_val_x = val_points.geometry.x.min() if train_val_test else None
    easternmost_val_x = val_points.geometry.x.max() if train_val_test else None
    westernmost_train_x = train_points.geometry.x.min()

    division_x_test = (easternmost_test_x + (westernmost_val_x if train_val_test else westernmost_train_x)) / 2
    division_x_val = (easternmost_val_x + westernmost_train_x) / 2 if train_val_test else None

    # Step 4: Load the TIFF file
    with rasterio.open(image_path) as src:
        # Convert the division_x to pixel coordinate
        division_col_test = int((division_x_test - src.bounds.left) / src.res[0])
        division_col_val = int((division_x_val - src.bounds.left) / src.res[0]) if train_val_test else None

        width = src.width
        height = src.height
        
        test_window = Window(0, 0, division_col_test, height)
        val_window = Window(division_col_test, 0, division_col_val - division_col_test, height) if train_val_test else None
        train_window = Window(division_col_val, 0, width - division_col_val, height) if train_val_test else Window(division_col_test, 0, width - division_col_test, height)

        # Read image parts
        test_read = src.read(window=test_window)
        val_read = src.read(window=val_window) if train_val_test else None
        train_read = src.read(window=train_window)
        
        if save_splits:
            os.makedirs(image_output_dir, exist_ok=True)
            os.makedirs(image_output_dir + "test/", exist_ok=True)
            os.makedirs(image_output_dir + "val/", exist_ok=True) if train_val_test else None
            os.makedirs(image_output_dir + "train/", exist_ok=True)
            
            # Save the western part
            test_output = image_output_dir + "test/" + f"{image_basename}_test.tif"

            # Save test image
            with rasterio.open(
                test_output,
                "w",
                driver="GTiff",
                height=test_window.height,
                width=test_window.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                nodata=-32767.0,
                transform=rasterio.windows.transform(test_window, src.transform),
            ) as dst:
                dst.write(test_read)

            # Save validation image
            if train_val_test:
                val_output = image_output_dir + "val/" + f"{image_basename}_val.tif"
                with rasterio.open(
                    val_output,
                    "w",
                    driver="GTiff",
                    height=val_window.height,
                    width=val_window.width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    nodata=-32767.0,
                    transform=rasterio.windows.transform(val_window, src.transform),
                ) as dst:
                    dst.write(val_read)
        
            # Save train image
            train_output = image_output_dir + "train/" + f"{image_basename}_train.tif"
            with rasterio.open(
                train_output,
                "w",
                driver="GTiff",
                height=train_window.height,
                width=train_window.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                nodata=-32767.0,
                transform=rasterio.windows.transform(train_window, src.transform),
            ) as dst:
                dst.write(train_read)
    
    # Step 5: Modify shapefile with labels
    def label_partition(x):
        if x < division_x_test:
            return "test"
        elif train_val_test and x < division_x_val:
            return "val"
        return "train"

    gdf["partition"] = gdf.x.apply(label_partition)

    if save_gdf_modified:
        os.makedirs(points_output_dir, exist_ok=True)
        # Step 6: Save the modified shapefile
        modified_shapefile_path = points_output_dir + f"{points_basename}.shp"
        gdf.to_file(modified_shapefile_path)

    if save_splits:
        print(f"Test{', validation,' if train_val_test else ''} and train TIFF parts saved in '{image_output_dir}'.")

    if save_gdf_modified:
        print(f"Modified shapefile saved as '{modified_shapefile_path}'.")
    
    return test_read, val_read, train_read, gdf

    
# def study_field_train_test_split(image_path, points_path, image_basename=None, image_output_dir=None, points_basename=None, points_output_dir=None, save_splits=True, save_gdf_modified=True, split=0.25):
#     """
#     Splits an image and its corresponding shapefile into training and testing parts based on spatial location.

#     Args:
#         image_path (str): Path to the input image (TIFF) file.
#         points_path (str): Path to the input shapefile containing points.
#         image_basename (str): Basename for saving output image files.
#         image_output_dir (str): Directory to save split image files.
#         points_basename (str): Basename for saving modified shapefile.
#         points_output_dir (str): Directory to save modified shapefile.
#         save_splits (bool, optional): If True, save split image files. Defaults to True.
#         save_gdf_modified (bool, optional): If True, save modified shapefile. Defaults to True.
#         split (float, optional): Fraction of points to assign to the test split, based on spatial position. Defaults to 0.25.

#     Returns:
#         tuple: (numpy.ndarray, numpy.ndarray, GeoDataFrame) - Arrays of the west (test) and east (train) parts of the image and the modified shapefile with "partition" column added.
#     """
#     # Step 1: Load the shapefile
#     gdf = gpd.read_file(points_path)

#     # Step 2: Identify the western 25% of points
#     # Sort by longitude (x-coordinate) and find the western 25%
#     gdf['x'] = gdf.geometry.x
#     gdf_sorted = gdf.sort_values(by="x")
#     num_western_points = int(np.ceil(len(gdf) * split))
#     western_points = gdf_sorted[:num_western_points]
#     remaining_points = gdf_sorted[num_western_points:]

#     # Step 3: Calculate division line
#     # Get x-coordinate of the easternmost point in the western group 
#     # and of the westernmost point in the remaining points
#     easternmost_west_x = western_points.geometry.x.max()
#     westernmost_east_x = remaining_points.geometry.x.min()
#     # Calculate midpoint x for division
#     division_x = (easternmost_west_x + westernmost_east_x) / 2

#     # Step 4: Load the TIFF file
#     with rasterio.open(image_path) as src:
#         # Convert the division_x to pixel coordinate
#         division_col = int((division_x - src.bounds.left) / src.res[0])
#         # print(division_col)
#         # Define windows for western and eastern parts
#         width = src.width
#         height = src.height
#         west_window = Window(0, 0, division_col, height)
#         east_window = Window(division_col, 0, width - division_col, height)

#         # Read the western part
#         west_read = src.read(window=west_window)
#         east_read = src.read(window=east_window)

#         if save_splits:
#             os.makedirs(image_output_dir, exist_ok=True)
#             os.makedirs(image_output_dir + "test/", exist_ok=True)
#             os.makedirs(image_output_dir + "train/", exist_ok=True)
            
#             # Save the western part
#             west_output = image_output_dir + "test/" + f"{image_basename}_test.tif"
            
#             with rasterio.open(
#                 west_output,
#                 "w",
#                 driver="GTiff",
#                 height=west_window.height,
#                 width=west_window.width,
#                 count=src.count,
#                 dtype=src.dtypes[0],
#                 crs=src.crs,
#                 nodata=-32767.0,
#                 transform=rasterio.windows.transform(west_window, src.transform),
#             ) as dst:
#                 dst.write(west_read)
        
#             # Save the eastern part
#             east_output = image_output_dir + "train/"  + f"{image_basename}_train.tif"
#             with rasterio.open(
#                 east_output,
#                 "w",
#                 driver="GTiff",
#                 height=east_window.height,
#                 width=east_window.width,
#                 count=src.count,
#                 dtype=src.dtypes[0],
#                 crs=src.crs,
#                 nodata=-32767.0,
#                 transform=rasterio.windows.transform(east_window, src.transform),
#             ) as dst:
#                 dst.write(east_read)
    
#     # Step 5: Modify shapefile with labels
#     gdf["partition"] = gdf.x.apply(lambda x: "test" if x < division_x else "train")

#     if save_gdf_modified:
#         os.makedirs(points_output_dir, exist_ok=True)
#         # Step 6: Save the modified shapefile
#         modified_shapefile_path = points_output_dir + f"{points_basename}.shp"
#         gdf.to_file(modified_shapefile_path)

#     if save_splits:
#         print(f"""Test and train TIFF parts saved as '{west_output}' and '{east_output}' respectively.""")

#     if save_gdf_modified:
#         print(f"""
#         Modified shapefile saved as '{modified_shapefile_path}'.""")
    
#     return west_read, east_read, gdf