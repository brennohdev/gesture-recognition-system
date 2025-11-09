import cv2
import numpy as np
import numpy.typing as npt
import typing

# --- Type Aliases ---
ImageType = npt.NDArray[np.uint8]
PointType = typing.Optional[tuple[int, int]]
ColorType = tuple[int, int, int]

# --- Constants ---
APP_NAME: str = "Virtual Paint"

WIDTH: int = 640
HEIGHT: int = 480

LOWER_BLUE: npt.NDArray[np.int32] = np.array([90, 50, 50])
UPPER_BLUE: npt.NDArray[np.int32] = np.array([140, 255, 255])

MIN_RADIUS: float = 10.0
BRUSH_THICKNESS: int = 8

COLOR_BRUSH: ColorType = (0, 0, 255)       # Red
COLOR_INDICATOR: ColorType = (0, 255, 255) # Yellow
COLOR_INDICATOR_DOT: ColorType = (0, 0, 255) # Red
COLOR_INDICATOR_RING: ColorType = (255, 255, 255) # White

HUD_BG_COLOR: ColorType = (0, 0, 0)
HUD_TEXT_COLOR_OK: ColorType = (0, 255, 0)     # Green
HUD_TEXT_COLOR_WARN: ColorType = (0, 0, 255)  # Red
FONT: int = cv2.FONT_HERSHEY_SIMPLEX

def main() -> None:
    print(f"🎨 {APP_NAME} - The Version That WILL Work!")
    print("=" * 60)

    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    ret: bool
    test_frame: ImageType
    ret, test_frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        return

    print(f"Original resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print(f"Working resolution: {WIDTH}x{HEIGHT}")
    print("=" * 60)
    print("Show a BLUE object to draw!")
    print("Press 'q' to quit, 'c' to clear\n")

    canvas: ImageType = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    prev_center: PointType = None
    frame_count: int = 0
    draw_count: int = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.flip(frame, 1)
        
        hsv: ImageType = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask: ImageType = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours: tuple
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center: PointType = None
        
        if len(contours) > 0:
            largest: npt.NDArray = max(contours, key=cv2.contourArea)
            
            x_f: float
            y_f: float
            radius_f: float
            ((x_f, y_f), radius_f) = cv2.minEnclosingCircle(largest)
            
            x: int = max(0, min(int(x_f), WIDTH - 1))
            y: int = max(0, min(int(y_f), HEIGHT - 1))
            
            if radius_f > MIN_RADIUS:
                center = (x, y)
                draw_count += 1
                
                cv2.circle(frame, center, int(radius_f), COLOR_INDICATOR, 5)
                cv2.circle(frame, center, 15, COLOR_INDICATOR_DOT, -1)
                cv2.circle(frame, center, 20, COLOR_INDICATOR_RING, 3)
                
                cv2.circle(canvas, center, BRUSH_THICKNESS, COLOR_BRUSH, -1)
                
                if prev_center:
                    cv2.line(canvas, prev_center, center, COLOR_BRUSH, BRUSH_THICKNESS)
                
                prev_center = center
                
                if frame_count % 30 == 0:
                    print(f"✅ Drawing at ({center[0]}, {center[1]}) | "
                          f"Total: {draw_count}")
        
        result: ImageType = cv2.addWeighted(frame, 1.0, canvas, 0.7, 0)
        
        cv2.rectangle(result, (0, 0), (WIDTH, 70), HUD_BG_COLOR, -1)
        
        cv2.putText(result, f"Drawings: {draw_count}", (10, 25),
                    FONT, 0.7, HUD_TEXT_COLOR_OK, 2)
        
        if center:
            cv2.putText(result, f"DRAWING! ({center[0]}, {center[1]})", 
                        (10, 50), FONT, 0.6, HUD_TEXT_COLOR_OK, 2)
        else:
            cv2.putText(result, "Looking for blue object...", (10, 50),
                        FONT, 0.6, HUD_TEXT_COLOR_WARN, 2)
        
        cv2.imshow(APP_NAME, result)
        cv2.imshow("Mask", mask)
        
        key: int = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            prev_center = None
            draw_count = 0
            print("🗑️  Canvas cleared!")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Total drawings: {draw_count}")

if __name__ == "__main__":
    main()