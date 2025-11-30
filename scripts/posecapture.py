import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple


def load_video(path: str) -> cv2.VideoCapture:
    """
    abre video a partir de caminho fornecido
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"nao foi possível abrir o vídeo: {path}")
    return cap


def get_video_properties(cap: cv2.VideoCapture) -> Tuple[int, int, int]:
    """
    a função obtém propriedades do video: largura, altura e fps.
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return width, height, fps


def create_video_writer(
    path: str, fps: int, frame_size: Tuple[int, int]
) -> cv2.VideoWriter:
    """
    basicamente cria um objeto VideoWriter para salvar o vídeo processado
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, frame_size)


def process_frame(
    frame: np.ndarray,
    pose,
    drawing_utils,
    landmark_color: Tuple[int, int, int] = (245, 117, 66),
    connection_color: Tuple[int, int, int] = (245, 66, 230),
) -> np.ndarray:
    """
    Processa um frame para detectar e desenhar landmarks de pose
    """
    # converter para RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # processar pose
    results = pose.process(image_rgb)

    # reconverter para BGR para desenhar
    image_rgb.flags.writeable = True
    output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # desenhar landmarks caso existam
    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            output_image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            drawing_utils.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
            drawing_utils.DrawingSpec(color=connection_color, thickness=2, circle_radius=2),
        )

    # inserir texto informativo
    cv2.putText(
        output_image,
        "SWISH.AI ENGINE v1.0",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return output_image


def main(input_video: str, output_video: str) -> None:
    """
    orquestra tudo
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = load_video(input_video)
    frame_width, frame_height, fps = get_video_properties(cap)

    writer = create_video_writer(output_video, fps, (frame_width, frame_height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, pose, mp_drawing)

            writer.write(processed_frame)

            cv2.imshow("Swish.AI - Analise de Pose", processed_frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("videoedu.mp4", "output_arremesso.mp4")
