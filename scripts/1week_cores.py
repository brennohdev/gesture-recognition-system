import cv2
import numpy as np

# --- 1. The Canvas ---
# Create a 3D numpy array: 600 rows (height), 600 columns (width), 3 channels (BGR)
# We use np.uint8, which means "unsigned 8-bit integer" (a number from 0-255).
# This is the standard data type for images.
# np.zeros creates an array filled with 0s, which gives us a black background.
image = np.zeros((600, 600, 3), dtype=np.uint8)

# --- 2. The Colors ---
# Let's define our colors as (B, G, R) tuples
# (Remember, it's BGR, not RGB)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# --- 3. Your Assignment: The Drawing ---
# All drawing functions "mutate" (draw on) the 'image' array in place.

# TODO: Draw a GREEN line
# cv2.line(image_to_draw_on, start_point_xy, end_point_xy, color, thickness)
# Let's draw a diagonal line from the top-left (0, 0) to (150, 150)
# cv2.line(...)
cv2.line(image, (0,0), (150, 150), GREEN, 5)

# TODO: Draw a BLUE rectangle
# cv2.rectangle(image, top_left_corner_xy, bottom_right_corner_xy, color, thickness)
# A thickness of -1 means "fill the shape". Let's make a filled one.
# Let's draw it from (150, 150) to (300, 300)
# cv2.rectangle(...)
cv2.rectangle(image, (150,150), (300, 300), BLUE, -1)

# TODO: Draw a RED circle
# cv2.circle(image, center_xy, radius, color, thickness)
# Let's put its center at (400, 400) with a radius of 50
# cv2.circle(...)
cv2.circle(image, (400,400), 50, RED, -1)

# TODO: Draw WHITE text
# cv2.putText(image, "My Text", origin_xy, font, font_scale, color, thickness)
# The "origin" is the bottom-left corner of the text.
font = cv2.FONT_HERSHEY_SIMPLEX
# Let's put the text "Clean Architecture" at (50, 500)
# cv2.putText(...)
cv2.putText(image, "Clean Architecture", (50, 500), font, 1, WHITE, 2)


# --- 4. The Display ---
# This is the "game loop" to show our image.
print("Showing image. Press 'q' to quit.")
while True:
    # Display the numpy array we've been drawing on
    cv2.imshow("My Drawing Canvas", image)
    
    # Wait for 1 millisecond. If the 'q' key is pressed, break the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Cleanup ---
# Close all the windows we opened
cv2.destroyAllWindows()
print("Script finished.")