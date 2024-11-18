import rasterio 
import numpy as np

def read_tiff(file_path : str) -> np.ndarray:
    """
    Load a geotiff file 
    """

    with rasterio.open(file_path) as src :
        data = src.read() 
    
    return data 