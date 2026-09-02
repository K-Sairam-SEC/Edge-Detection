edge-detection-opencv
## Aim
To perform edge detection using Sobel, Roberts, Prewitt, Laplacian, and Canny edge detectors.

## Software Required
Anaconda – Python 3.7
Jupyter Notebook / VS Code
OpenCV (cv2)
NumPy
Matplotlib
⚙️ Algorithm
# Step 1:
Import all the necessary modules for the program.

# Step 2:
Load an image using cv2.imread().

# Step 3:
Convert the image to grayscale.

# Step 4:
Apply Sobel operator using OpenCV to detect edges.

# Step 5:
Apply Prewitt operator using custom kernels.

# Step 6:
Apply Roberts operator using custom kernels.

# Step 7:
Apply Laplacian operator using OpenCV.

# Step 8:
Apply Canny edge detector using OpenCV.

# Step 9:
Display all edge-detected images for comparison.

# Developed By
Name: Sairam K
Register No: 212225240132

# Program
```py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display original image
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis("off")
plt.show()


sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel = cv2.magnitude(sx, sy)

plt.imshow(sobel, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis("off")
plt.show()


px = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])

py = np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]])

prewitt_x = cv2.filter2D(gray, -1, px)
prewitt_y = cv2.filter2D(gray, -1, py)

prewitt = cv2.magnitude(prewitt_x.astype(np.float32),
                        prewitt_y.astype(np.float32))

plt.imshow(prewitt, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis("off")
plt.show()


rx = np.array([[1, 0],
               [0, -1]])

ry = np.array([[0, 1],
               [-1, 0]])

roberts_x = cv2.filter2D(gray, -1, rx)
roberts_y = cv2.filter2D(gray, -1, ry)

roberts = cv2.magnitude(roberts_x.astype(np.float32),
                        roberts_y.astype(np.float32))

plt.imshow(roberts, cmap='gray')
plt.title("Roberts Edge Detection")
plt.axis("off")
plt.show()


laplacian = cv2.Laplacian(gray, cv2.CV_64F)

plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Edge Detection")
plt.axis("off")
plt.show()


canny = cv2.Canny(gray, 100, 200)

plt.imshow(canny, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()
```
# Output
<img width="272" height="412" alt="image" src="https://github.com/user-attachments/assets/f0e5c129-6e89-4576-a526-c553aeb257e5" />
<img width="275" height="405" alt="image" src="https://github.com/user-attachments/assets/8934c0e2-e931-400f-9c13-c48bcfde4ca1" />
<img width="264" height="405" alt="image" src="https://github.com/user-attachments/assets/cd2f4934-84fc-4911-b439-cfa986b8c318" />
<img width="291" height="409" alt="image" src="https://github.com/user-attachments/assets/68455ea5-0587-47cf-bc77-6ad2ed27f3e7" />
<img width="272" height="407" alt="image" src="https://github.com/user-attachments/assets/8acf762d-745f-4207-a84c-1946b9b86688" />

# Sobel Edge Detector
Detects edges in horizontal and vertical directions
Produces gradient-based edge map
# Prewitt Edge Detector
Similar to Sobel but simpler kernel
Detects directional edges
# Roberts Edge Detector
Detects edges using diagonal gradients
Sensitive to noise
# Laplacian Edge Detector
Detects edges using second-order derivatives
Highlights rapid intensity changes
# Canny Edge Detector
Multi-stage edge detection
Produces clean and thin edges

Result
Thus, edges are successfully detected using Sobel, Prewitt, Roberts, Laplacian, and Canny edge detection techniques. Each method highlights edges differently based on gradient and intensity variations, improving feature extraction and analysis.
