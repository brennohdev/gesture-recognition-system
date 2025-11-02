import cv2

def main():
    print("🎥 Abrindo câmera...")
    print("Pressione 'q' para sair")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir câmera!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, "Hello OpenCV! (q = sair)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        h, w = frame.shape[:2]
        cv2.circle(frame, (w//2, h//2), 50, (0, 0, 255), 3)
        
        cv2.imshow("Hello OpenCV", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finalizado!")

if __name__ == "__main__":
    main()