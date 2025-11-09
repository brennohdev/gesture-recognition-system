import cv2
import numpy as np
import numpy.typing as npt

# A type alias for our images to keep things clean
ImageType = npt.NDArray[np.uint8]

def main() -> None:
    # 1. SETUP: Get the camera
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Looking for BLUE objects. Press 'q' to quit.")

    while True:
        # 2. READ: Get a frame
        ret: bool
        frame: ImageType
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for that natural mirror-effect
        frame = cv2.flip(frame, 1)

        # --- THIS IS THE NEW PART ---

        # 3. CONVERT: BGR -> HSV
        # We create a new version of the frame in HSV color space
        hsv_frame: ImageType = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 4. FILTER (MASK): Define the color range
        # We need a 'lower' and 'upper' bound for "blue"
        # These are [H, S, V] arrays
        # Hue: ~100-130 is blue in OpenCV's 0-179 range
        # Saturation: 150-255 (we want a pretty pure, not-gray color)
        # Value: 50-255 (we want a somewhat bright color, not black)
        lower_blue: npt.NDArray[np.int32] = np.array([100, 150, 50])
        upper_blue: npt.NDArray[np.int32] = np.array([130, 255, 255])

        # This is the magic line.
        # It creates a new image (mask) that is:
        # - WHITE (255) where pixels in hsv_frame were WITHIN the range
        # - BLACK (0) where pixels were OUTSIDE the range
        mask: ImageType = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # 5. APPLY: Use the mask on the ORIGINAL frame
        # cv2.bitwise_and(src1, src2, mask=mask)
        # This says: "Perform a 'bitwise AND' between 'frame' and 'frame',
        # BUT only in the areas where the 'mask' is white (not 0)."
        # The result: The original pixel is kept if the mask was white.
        #            The pixel is set to black (0) if the mask was black.
        result: ImageType = cv2.bitwise_and(frame, frame, mask=mask)

        # --- END OF NEW PART ---

        # 6. DISPLAY: Show all three windows to see what's happening
        cv2.imshow("1. Original Frame", frame)
        cv2.imshow("2. The Mask (Our Stencil)", mask)
        cv2.imshow("3. Filtered Result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. CLEANUP
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")

if __name__ == "__main__":
    main()