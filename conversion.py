from scipy.io import loadmat, savemat
import rasterio

# Read TIFF file
tiff_path = "./data/vel.mskd.geo.tif"
with rasterio.open(tiff_path) as src:
    data = src.read()  # Shape: (bands, height, width)
    profile = src.profile  # Metadata

# Prepare dictionary for MATLAB file
mat_data = {
    "satellite_data": data,      # The actual pixel values
    "metadata": profile          # Optional: metadata about the raster
}
# Save as MAT file
savemat("data/datasets/tests/output.mat", mat_data)

print("Conversion complete: output.mat")
Dictionary = loadmat('data/datasets/tests/output.mat')
print(Dictionary)