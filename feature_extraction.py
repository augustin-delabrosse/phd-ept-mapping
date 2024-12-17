import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt
from rasterio.transform import from_origin
# from rasterio.enums import Resampling
import os
from scipy.ndimage import binary_opening, binary_closing
from pyproj import Transformer
from rasterio.warp import calculate_default_transform, reproject, Resampling
import osmnx as ox

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from remote_sensing_utils import vector_to_raster



# def create_water_mask(image_path, water_path, output_dir, site, save_mask):
#     """
#     Creates a binary water mask from vector data (water zones and rivers) and saves it as a raster.

#     Parameters:
#     - image_path (str): Path to the multispectral image file.
#     - water_path (str): Path to the GeoPackage (.gpkg) file containing water zones and river data.
#     - output_dir (str): Directory where the water mask will be saved.
#     - site (str): Site identifier to filter the vector data (Capital first letter, no accent Ex. "Timbertiere")
#     - save_mask (bool): Whether to save the generated water mask to disk.

#     Returns:
#     - water_raster (numpy.ndarray): The generated binary water mask.
#     - height (int): Height of the raster (in pixels).
#     - width (int): Width of the raster (in pixels).
#     - crs (CRS): Coordinate reference system of the raster.
#     - transform (Affine): Affine transformation of the raster.
#     """
#     # Load multispectral image to get spatial reference and metadata
#     with rasterio.open(image_path) as src:
#         profile = src.profile
#         transform = src.transform
#         width = src.width
#         height = src.height
#         crs = src.crs
    
#     # Load water zones vector data and filter by site
#     water_gdf = gpd.read_file(water_path, layer="Surface en eau")
#     water_gdf = water_gdf[water_gdf['Site'] == site]
    
#     # Load rivers vector data and filter by site
#     rivers_gdf = gpd.read_file(water_path, layer="reseauhydro_lineaire")
#     rivers_gdf = rivers_gdf[rivers_gdf['Site'] == site] 

#     # Create a binary water raster using the vector data
#     water_raster = vector_to_raster([water_gdf, rivers_gdf], transform, width, height)

#     if save_mask:
#         # Write the binary water mask raster to file
#         with rasterio.open(
#             os.path.join(output_dir, f'water_mask_{str(site).lower()}.tif'), 
#             'w', driver='GTiff', height=height, width=width,
#             count=1, dtype='uint8', crs=crs, transform=transform) as dst:
#             dst.write(water_raster, 1)
#             # dst.set_nodata(-9999)  # Optional: set a nodata value
        
#         print("Water raster has been created and saved to", output_dir)

#     return water_raster, height, width, crs, transform

# def create_water_distance_map(image_path, water_path, output_dir, site, save_mask):
#     """
#     Creates a distance map from the water mask, showing the distance of each pixel to the nearest water body.

#     Parameters:
#     - image_path (str): Path to the multispectral image file.
#     - water_path (str): Path to the GeoPackage (.gpkg) file containing water zones and river data.
#     - output_dir (str): Directory where the distance map will be saved.
#     - site (str): Site identifier to filter the vector data (Capital first letter, no accent Ex. "Timbertiere").
#     - save_mask (bool): Whether to save the generated water mask before creating the distance map.

#     Returns:
#     - distance_raster (numpy.ndarray): The distance map raster showing pixel distances to the nearest water body.
#     """
#     # Create the water mask from vector data
#     water_raster, height, width, crs, transform = create_water_mask(image_path, water_path, output_dir, site, save_mask)
    
#     # Create a distance map where each pixel value represents the distance to the nearest water pixel
#     distance_raster = distance_transform_edt(water_raster == 0)  # Calculates the Euclidean distance to the nearest non-water pixel
        
#     # Write the distance raster to file
#     with rasterio.open(
#         os.path.join(output_dir, f'water_distance_map_{str(site).lower()}.tif'), 
#         'w', driver='GTiff', height=height, width=width,
#         count=1, dtype='float32', crs=crs, transform=transform) as dst:
#         dst.write(distance_raster, 1)
#         # dst.set_nodata(-9999)  # Optional: set a nodata value
    
#     print("Distance raster has been created and saved to", output_dir)

#     return distance_raster

class WaterDistanceMap:
    def __init__(self, image_path, water_path):
        """
        Initialize the WaterDistanceMap class.

        Parameters:
        - image_path (str): Path to the multispectral image file.
        - water_path (str): Path to the GeoPackage (.gpkg) file containing water zones and river data.
        """
        self.image_path = image_path
        self.water_path = water_path

    @staticmethod
    def add_buffer_to_geometry(gdf):
        """
        Adds a buffer to the geometry column of a GeoDataFrame based on the values in the 'Largeur/2' column.
        If the 'Largeur/2' value is NaN, no buffer is applied.
    
        Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame containing the geometry and 'Largeur/2' columns.
    
        Returns:
        GeoDataFrame: A new GeoDataFrame with the modified geometries.
        """
        # Ensure the 'Largeur/2' column exists and is numeric
        if 'Largeur/2' not in gdf.columns:
            raise ValueError("Column 'Largeur/2' is missing in the GeoDataFrame.")
        
        # Apply buffer to each geometry, skipping rows with NaN in 'Largeur/2'
        gdf['geometry'] = gdf.apply(
            lambda row: row['geometry'].buffer(row['Largeur/2']) if not np.isnan(row['Largeur/2']) else row['geometry'],
            axis=1
        )
        
        return gdf
        
    def create_water_mask(self, site, output_dir=None, save_mask=True):
        """
        Creates a binary water mask from vector data (water zones and rivers) and saves it as a raster.

        Parameters:
        - output_dir (str): Directory where the water mask will be saved.
        - site (str): Site identifier to filter the vector data (Capital first letter, no accent Ex. "Timbertiere")
        - save_mask (bool): Whether to save the generated water mask to disk.

        Returns:
        - water_raster (numpy.ndarray): The generated binary water mask.
        - height (int): Height of the raster (in pixels).
        - width (int): Width of the raster (in pixels).
        - crs (CRS): Coordinate reference system of the raster.
        - transform (Affine): Affine transformation of the raster.
        """
        # Load multispectral image to get spatial reference and metadata
        with rasterio.open(self.image_path) as src:
            profile = src.profile
            transform = src.transform
            width = src.width
            height = src.height
            crs = src.crs

        # Load water zones vector data and filter by site
        water_gdf = gpd.read_file(self.water_path, layer="Surface en eau")
        water_gdf = water_gdf[water_gdf['Site'] == site]

        # Load rivers vector data and filter by site
        rivers_gdf = gpd.read_file(self.water_path, layer="reseauhydro_lineaire")
        rivers_gdf = rivers_gdf[rivers_gdf['Site'] == site]
        rivers_gdf = self.add_buffer_to_geometry(rivers_gdf)

        # Create a binary water raster using the vector data
        water_raster = vector_to_raster([water_gdf, rivers_gdf], transform, width, height)

        if save_mask:
            # Write the binary water mask raster to file
            mask_path = os.path.join(output_dir, f'water_mask_{site.lower()}.tif')
            with rasterio.open(mask_path, 'w', driver='GTiff', height=height, width=width,
                               count=1, dtype='uint8', crs=crs, transform=transform) as dst:
                dst.write(water_raster, 1)
            print(f"Water raster has been created and saved to {mask_path}")

        return water_raster, height, width, crs, transform

    def create_water_distance_map(self, site, output_dir, save_mask=True):
        """
        Creates a distance map from the water mask, showing the distance of each pixel to the nearest water body.

        Parameters:
        - output_dir (str): Directory where the distance map will be saved.
        - site (str): Site identifier to filter the vector data (Capital first letter, no accent Ex. "Timbertiere").
        - save_mask (bool): Whether to save the generated water mask before creating the distance map.

        Returns:
        - distance_raster (numpy.ndarray): The distance map raster showing pixel distances to the nearest water body.
        """
        # Create the water mask from vector data
        water_raster, height, width, crs, transform = self.create_water_mask(site, output_dir, save_mask)

        # Create a distance map where each pixel value represents the distance to the nearest water pixel
        distance_raster = distance_transform_edt(water_raster == 0)  # Calculates the Euclidean distance to the nearest non-water pixel

        # Write the distance raster to file
        distance_map_path = os.path.join(output_dir, f'water_distance_map_{site.lower()}.tif')
        with rasterio.open(distance_map_path, 'w', driver='GTiff', height=height, width=width,
                           count=1, dtype='float32', crs=crs, transform=transform) as dst:
            dst.write(distance_raster, 1)

        print(f"Distance raster has been created and saved to {distance_map_path}")

        return distance_raster



# def load_align_image_and_dhm(image_path, dhm_path):
#     # Load the multispectral image
#     with rasterio.open(image_path) as src:
#         multispectral_image = src.read()
#         multispectral_transform = src.transform
#         multispectral_crs = src.crs
#         multispectral_bounds = src.bounds
#         multispectral_profile = src.profile
#         multispectral_width = src.width
#         multispectral_height = src.height
#         multispectral_meta = src.meta
    
#     # Load the DHM
#     with rasterio.open(dhm_path) as dhm_src:
#         dhm = dhm_src.read(1)
#         dhm_transform = dhm_src.transform
#         dhm_crs = dhm_src.crs

#     # Manually set the DHM CRS to match the multispectral image CRS
#     dhm_crs_forced = multispectral_crs
    
#     # Clip or resample the DHM to match the multispectral image
#     dhm_aligned = np.empty((multispectral_image.shape[1], multispectral_image.shape[2]), dtype=rasterio.float32)
    
#     # Calculate the transform and bounds of the multispectral image
#     out_transform, out_width, out_height = calculate_default_transform(
#         multispectral_crs, multispectral_crs, multispectral_image.shape[2], multispectral_image.shape[1],
#         *multispectral_bounds)
    
#     # Reproject DHM to match the multispectral image dimensions
#     reproject(
#         source=dhm,
#         destination=dhm_aligned,
#         src_transform=dhm_transform,
#         src_crs=dhm_crs_forced,
#         dst_transform=out_transform,
#         dst_crs=multispectral_crs,
#         resampling=Resampling.bilinear
#     )
    
#     # Now `dhm_aligned` should match the size and alignment of your multispectral image
#     print(f"DHM aligned shape: {dhm_aligned.shape}")
#     print(f"Multispectral image shape: {multispectral_image.shape[1:]}")

#     return multispectral_image, dhm_aligned, multispectral_transform, multispectral_width, multispectral_height, multispectral_bounds, multispectral_meta, multispectral_crs


# def create_NDVI(multispectral_image):
#     # Extract Red and NIR bands
#     red_band = multispectral_image[5]
#     nir_band = multispectral_image[9]
    
#     # Compute NDVI
#     ndvi = (nir_band.astype(float) - red_band.astype(float)) / (nir_band + red_band)
#     return ndvi


# def load_buildings(multispectral_bounds, multispectral_crs, src_crs='EPSG:2154', target_crs='EPSG:4326'):
#     # Create a transformer object
#     transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

#     min_lon, min_lat = transformer.transform(multispectral_bounds.left, multispectral_bounds.bottom)
#     max_lon, max_lat = transformer.transform(multispectral_bounds.right, multispectral_bounds.top)
    
#     # Fetch OSM building footprints for the area of interest
#     gdf_buildings = ox.geometries_from_bbox(north=max_lat, south=min_lat, east=max_lon, west=min_lon,
#                                         tags={'building': True})

#     if gdf_buildings.crs != multispectral_crs:
#         gdf_buildings = gdf_buildings.to_crs(multispectral_crs)
#     gdf_buildings = gdf_buildings.buffer(3)

#     return gdf_buildings

# def create_hedgerow_mask(image_path, dhm_path, ndvi_threshold=0.3, height_threshold=2.5):

#     multispectral_image, dhm_aligned, multispectral_transform, multispectral_width, multispectral_height, multispectral_bounds, multispectral_meta, multispectral_crs = load_align_image_and_dhm(image_path, dhm_path)

#     ndvi = create_NDVI(multispectral_image)

#     gdf_buildings = load_buildings(multispectral_bounds, multispectral_crs)

#     # Create a vegetation mask using NDVI and DHM    
#     vegetation_mask = (ndvi > ndvi_threshold) & (dhm_aligned > height_threshold)

#     # Perform opening to remove small objects
#     cleaned_mask_1 = binary_opening(vegetation_mask, structure=np.ones((3, 3)))
    
#     # Perform closing to fill small holes
#     cleaned_mask_2 = binary_closing(cleaned_mask_1, structure=np.ones((3, 3)))

#     # Create a water raster
#     buildings_mask = vector_to_raster([gdf_buildings], multispectral_transform, multispectral_width, multispectral_height)

#     # Combine masks: Set vegetation areas overlapping with buildings to 0
#     final_mask = np.where(buildings_mask, 0, cleaned_mask_2)

#     return final_mask, multispectral_meta

# def save_mask(mask, meta, mask_path):
#     # Save the cleaned mask
#     out_meta = meta.copy()
#     out_meta.update({"driver": "GTiff", "count": 1, "dtype": 'uint8'})
    
#     # Remove or adjust nodata value
#     if 'nodata' in out_meta:
#         out_meta.pop('nodata') 
    
#     with rasterio.open(mask_path, "w", **out_meta) as dest:
#         dest.write(mask.astype(np.uint8), 1)

#     print("Mask has been created and saved to", mask_path)


class HedgerowMask:
    def __init__(self, image_path, dhm_path):
        """
        Initialize the HedgerowMask class.

        Parameters:
        - image_path (str): Path to the multispectral image file.
        - dhm_path (str): Path to the Digital Height Model (DHM) file.
        """
        self.image_path = image_path
        self.dhm_path = dhm_path

    def load_align_image_and_dhm(self):
        """
        Load and align the multispectral image and DHM to ensure they have the same dimensions and alignment.

        Returns:
        - multispectral_image (numpy.ndarray): The multispectral image array.
        - dhm_aligned (numpy.ndarray): The aligned DHM array.
        - multispectral_transform (Affine): The affine transform of the multispectral image.
        - multispectral_width (int): The width of the multispectral image.
        - multispectral_height (int): The height of the multispectral image.
        - multispectral_bounds (BoundingBox): The bounding box of the multispectral image.
        - multispectral_meta (dict): Metadata of the multispectral image.
        - multispectral_crs (CRS): Coordinate reference system of the multispectral image.
        """
        # Load the multispectral image
        with rasterio.open(self.image_path) as src:
            multispectral_image = src.read()
            multispectral_transform = src.transform
            multispectral_crs = src.crs
            multispectral_bounds = src.bounds
            multispectral_meta = src.meta
            multispectral_width = src.width
            multispectral_height = src.height

        # Load the DHM
        with rasterio.open(self.dhm_path) as dhm_src:
            dhm = dhm_src.read(1)
            dhm_transform = dhm_src.transform
            dhm_crs = dhm_src.crs

        # Manually set the DHM CRS to match the multispectral image CRS
        dhm_crs_forced = multispectral_crs

        # Clip or resample the DHM to match the multispectral image
        dhm_aligned = np.empty((multispectral_height, multispectral_width), dtype=rasterio.float32)

        # Calculate the transform and bounds of the multispectral image
        out_transform, out_width, out_height = calculate_default_transform(
            multispectral_crs, multispectral_crs, multispectral_width, multispectral_height,
            *multispectral_bounds)

        # Reproject DHM to match the multispectral image dimensions
        reproject(
            source=dhm,
            destination=dhm_aligned,
            src_transform=dhm_transform,
            src_crs=dhm_crs_forced,
            dst_transform=out_transform,
            dst_crs=multispectral_crs,
            resampling=Resampling.bilinear
        )

        return (multispectral_image, dhm_aligned, multispectral_transform, 
                multispectral_width, multispectral_height, multispectral_bounds, 
                multispectral_meta, multispectral_crs)

    def create_NDVI(self, multispectral_image):
        """
        Calculate the Normalized Difference Vegetation Index (NDVI) using the multispectral image.

        Parameters:
        - multispectral_image (numpy.ndarray): The multispectral image array.

        Returns:
        - ndvi (numpy.ndarray): The NDVI array.
        """
        # Extract Red and NIR bands
        red_band = multispectral_image[5]
        nir_band = multispectral_image[9]

        # Compute NDVI
        ndvi = (nir_band.astype(float) - red_band.astype(float)) / (nir_band + red_band)
        return ndvi

    def load_buildings(self, multispectral_bounds, multispectral_crs, src_crs='EPSG:2154', target_crs='EPSG:4326'):
        """
        Load building footprints from OpenStreetMap within the multispectral image bounds.

        Parameters:
        - multispectral_bounds (BoundingBox): The bounding box of the multispectral image.
        - multispectral_crs (CRS): Coordinate reference system of the multispectral image.
        - src_crs (str): Source CRS for the bounding box transformation.
        - target_crs (str): Target CRS for the bounding box transformation.

        Returns:
        - gdf_buildings (GeoDataFrame): GeoDataFrame containing building geometries.
        """
        # Create a transformer object
        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

        min_lon, min_lat = transformer.transform(multispectral_bounds.left, multispectral_bounds.bottom)
        max_lon, max_lat = transformer.transform(multispectral_bounds.right, multispectral_bounds.top)

        # Fetch OSM building footprints for the area of interest
        gdf_buildings = ox.geometries_from_bbox(north=max_lat, south=min_lat, east=max_lon, west=min_lon,
                                                tags={'building': True})

        # Reproject buildings to match the multispectral image CRS
        if gdf_buildings.crs != multispectral_crs:
            gdf_buildings = gdf_buildings.to_crs(multispectral_crs)
        gdf_buildings = gdf_buildings.buffer(3)  # Buffer to account for minor misalignments

        return gdf_buildings

    def create_hedgerow_mask(self, ndvi_threshold=0.3, height_threshold=2.5):
        """
        Create a hedgerow mask by combining NDVI and DHM data and excluding building areas.

        Parameters:
        - ndvi_threshold (float): NDVI threshold to identify vegetation.
        - height_threshold (float): Height threshold to distinguish hedgerows.

        Returns:
        - final_mask (numpy.ndarray): The generated hedgerow mask.
        - multispectral_meta (dict): Metadata of the multispectral image for saving the mask.
        """
        # Load and align image and DHM
        (multispectral_image, dhm_aligned, multispectral_transform, 
         multispectral_width, multispectral_height, multispectral_bounds, 
         multispectral_meta, multispectral_crs) = self.load_align_image_and_dhm()

        # Compute NDVI
        ndvi = self.create_NDVI(multispectral_image)

        # Load buildings from OSM
        gdf_buildings = self.load_buildings(multispectral_bounds, multispectral_crs)

        # Create a vegetation mask using NDVI and DHM    
        vegetation_mask = (ndvi > ndvi_threshold) & (dhm_aligned > height_threshold)

        # Perform opening to remove small objects and closing to fill small holes
        cleaned_mask_1 = binary_opening(vegetation_mask, structure=np.ones((3, 3)))
        cleaned_mask_2 = binary_closing(cleaned_mask_1, structure=np.ones((3, 3)))

        # Create a building raster
        buildings_mask = vector_to_raster([gdf_buildings], multispectral_transform, multispectral_width, multispectral_height)

        # Combine masks: Set vegetation areas overlapping with buildings to 0
        final_mask = np.where(buildings_mask, 0, cleaned_mask_2)

        return final_mask, multispectral_meta

    def save_mask(self, mask, meta, mask_path):
        """
        Save the cleaned mask as a GeoTIFF file.

        Parameters:
        - mask (numpy.ndarray): The mask array to be saved.
        - meta (dict): Metadata of the multispectral image.
        - mask_path (str): File path where the mask will be saved.
        """
        # Update metadata to match the mask
        out_meta = meta.copy()
        out_meta.update({"driver": "GTiff", "count": 1, "dtype": 'uint8'})

        # Remove or adjust nodata value
        if 'nodata' in out_meta:
            out_meta.pop('nodata')

        with rasterio.open(mask_path, "w", **out_meta) as dest:
            dest.write(mask.astype(np.uint8), 1)

        print(f"Mask has been created and saved to {mask_path}")
