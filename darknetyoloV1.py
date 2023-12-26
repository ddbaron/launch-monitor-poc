import cv2
import darknet

def process_video(input_video_path, output_video_path, darknet_config, darknet_weights, darknet_data):
    network, class_names, class_colors = darknet.load_network(
        darknet_config,
        darknet_data,
        darknet_weights,
        batch_size=1
    )
    video = cv2.VideoCapture(input_video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, codec, 30.0, (width, height))

    while True:
        ret, frame = video.read()
        if not ret:
            break
        darknet_image = darknet.make_image(width, height, 3)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
        darknet.free_image(darknet_image)

        for label, confidence, bbox in detections:
            if label == 'ball':
                x, y, w, h = bbox
                x1, y1, x2, y2 = darknet.bbox2points(bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors[class_names.index(label)], 2)

        out.write(frame)

    video.release()
    out.release()

if __name__ == "__main__":
    process_video('input.h264', 'output.avi', 'cfg/yolov4.cfg', 'yolov4.weights', 'cfg/coco.data')
