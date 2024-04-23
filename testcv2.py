import cv2
import numpy as np

# Create a blank 256x256 pixel image with 3 channels (RGB), and a black background
image = np.zeros((256, 256, 3), dtype=np.uint8)

# Draw a green rectangle
cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1)

# Show the image in a window
cv2.imshow('Image', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

# Save the image to a file
cv2.imwrite('output.png', image)
