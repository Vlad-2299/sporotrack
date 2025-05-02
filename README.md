# SPOROTRACKER

Analyze microscopy imagery to detect and track elliptical or circular bodies such as sporozoites.

![Tracked](assets/tracked.png)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/sporotracker.git
   ```

2. Navigate to the project directory:
   ```bash
   cd sporotracker
   ```

3. Run the setup script:
   - **Mac/Linux**: `./setup.sh`
   - **Windows**: `./setup.bat`

4. Launch the program:
   ```bash
   uv run main.py
   ```

## Frame Processing Pipeline

SPOROTRACKER employs a sophisticated image processing pipeline to identify and track microscopic bodies:

1. **Image Normalization**
   - Standardizes 8-bit frame values to the full 0-255 range
   - Ensures consistent brightness and contrast across frames

2. **Histogram Equalization**
   - Enhances contrast in high-intensity regions
   - Improves visibility of faint structures

   ![HistEq](assets/histeq.png)

3. **Threshold Masking** *(optional)*
   - Reduces noise by applying a pixel intensity threshold
   - Preserves values above threshold, sets values below to zero

   ![Mask](assets/mask.png)

4. **Bilateral Filtering**
   - Smooths image while preserving edge sharpness
   - Reduces noise without blurring important features

5. **Edge Detection**
   - Generates binary image with continuous edge lines
   - Highlights areas of high pixel intensity change

   ![Edges](assets/edges.png)

6. **Morphological Transformations**
   - Performs dilation followed by erosion
   - Fills small gaps between edges to create continuous boundaries

7. **Contour Detection**
   - Traces continuous points along object boundaries
   - Identifies potential sporozoites or other bodies of interest

   ![Contours](assets/contours.png)

## Program Features

SPOROTRACKER offers several analysis modes:

1. **Detect & Track Sporozoites**
   - Runs the complete processing pipeline
   - Identifies and follows sporozoites across frames
   - Displays tracking data in real-time

2. **Apply Histogram Equalization**
   - Shows results after step 2 of the pipeline
   - Helps evaluate contrast enhancement

3. **Apply Threshold Masking**
   - Shows results after step 3 of the pipeline
   - Aids in threshold adjustment

4. **Detect Edges**
   - Shows results after step 5 of the pipeline
   - Highlights boundary detection

5. **Detect Contours**
   - Shows results after step 7 of the pipeline
   - Visualizes identified bodies

6. **Detect All Bodies**
   - Identifies all elliptical shapes
   - Displays individual area measurements

   ![Detections](assets/detects.png)

7. **Trim TIF** *(Coming Soon)*
   - Functionality to trim TIF image sequences

8. **Crop TIF** *(Coming Soon)*
   - Functionality to crop TIF image sequences

9. **View Settings** *(Coming Soon)*
   - Configuration interface for processing parameters