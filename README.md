# SPOROTRACKER
Process microscopy imagery, detect & track elliptical, or circular, bodies.


## Installation
Use `git clone` to download the project.

To get started, in the `/sporotrack` directory, run:

- Mac/Linux: `./setup.sh`

- Windows: `./setup.bat`

Once the installation is complete, run the program with `uv run main.py`
---

## Frame processing pipeline

1. **Image Normalization:** Forces the 8bit frame values to always range between 0-255

2. **Image Histogram Equalization:** Helps to improve the contrast of the 8bit frame, only in the high pixel value rigions

3. **Image Masking (_optional_):** Helps to reduce noise by thresholding the 8bit frame, where all the pixel values above the threshold are preserved, and all the pixel values below the treshold are set to 0

4. **Image Smoothing:** Employs bilateral filter to reduce noise, while preserving edge sharpness

5. **Image Edge Detection:** Returns a binary image with continuous edge lines that represent  high pixel value intensity changes

6. **Morphological Transformations:** Dilation followed by erosion to fill small gaps beteen edges

7. **Contour Detection:** Traces the continuous points along object (edge) boundaries

