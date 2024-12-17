import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import box

def resample_dem_to_match(src_path, 
                          dest_transform, dest_crs,
                          dest_width, dest_height, 
                          same_crs=True, save_resampled=False, 
                          resample_output_dir=None, 
                          resample_output_file=None):
    """
    Resample the DEM to match the multispectral image's resolution and extent.
    
    Parameters:
    - src_path: str, Path to the source DEM image.
    - dest_transform: Affine, Geotransformation matrix for the destination image.
    - dest_crs: CRS, Coordinate Reference System for the destination image.
    - dest_width: int, Width of the destination image.
    - dest_height: int, Height of the destination image.
    - same_crs: bool, Whether to use the same CRS as the source image.
    - save_resampled: bool, If True, save the resampled DEM to a file.
    
    Returns:
    - resampled: np.ndarray, Resampled DEM data as a numpy array.
    """
    with rasterio.open(src_path) as src:
        src_bounds = src.bounds
        src_transform = src.transform
        src_crs = src.crs
        src_profile = src.profile

        if same_crs:
            dest_crs = src_crs

        intersection_bounds = (
            max(src_bounds.left, dest_transform[2]),  
            max(src_bounds.bottom, dest_transform[5]),  
            min(src_bounds.right, dest_transform[2] + dest_width * dest_transform[0]),  
            min(src_bounds.top, dest_transform[5] + dest_height * dest_transform[4])  
        )

        new_transform, new_width, new_height = calculate_default_transform(
            src_crs, dest_crs, dest_width, dest_height, *intersection_bounds
        )

        resampled = np.empty((new_height, new_width), dtype=src.read(1).dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=resampled,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=new_transform,
            dst_crs=dest_crs,
            resampling=Resampling.bilinear
        )

        src_profile.update({
            'transform': new_transform,
            'width': new_width,
            'height': new_height,
            'crs': dest_crs
        })

    if save_resampled and resample_output_dir:
        os.makedirs(resample_output_dir, exist_ok=True)
        output_path = os.path.join(resample_output_dir, resample_output_file)
        with rasterio.open(output_path, 'w', **src_profile) as dst:
            dst.write(resampled, 1)

    return resampled

def clip_vectors_to_image(gdf, image_bounds):
    """
    Clip vector data to the bounds of the image.
    
    Parameters:
    - gdf: GeoDataFrame, Input vector geometries (either line or polygon).
    - image_bounds: tuple, Bounding box of the raster image.
    
    Returns:
    - GeoDataFrame: Clipped vector geometries.
    """
    return gdf.clip(box(*image_bounds))
    

def vector_to_raster(gdfs, transform, width, height):
    """
    Converts vector data (geometries) into a binary raster.

    Parameters:
    - gdfs (list of GeoDataFrames): List of GeoDataFrames containing the geometries (e.g., polygons, lines) to be rasterized.
    - transform (Affine): Affine transformation object that defines the mapping of raster space to geographic space.
    - width (int): Width of the output raster (number of pixels).
    - height (int): Height of the output raster (number of pixels).

    Returns:
    - raster (numpy.ndarray): Binary raster (2D array) where pixels corresponding to geometries are set to 1, and others to 0.
    """
    # Initialize an empty raster array with zeros (background)
    raster = np.zeros((height, width), dtype=np.uint8)
    
    # Loop through each GeoDataFrame to rasterize its geometries
    for gdf in gdfs:
        for geom in gdf.geometry:
            # Create a mask of the geometry, setting raster pixels to True where geometry is present
            mask = geometry_mask([geom], transform=transform, invert=True, out_shape=(height, width))
            # Set the corresponding pixels in the raster array to 1
            raster[mask] = 1
    
    return raster