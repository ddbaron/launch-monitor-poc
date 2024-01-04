import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_offset(res_w=1456, res_h=1088, crop_w=1152, crop_h=192, centroid_x=None, centroid_y=None):
    """
    Calculate the region of interest (ROI) based on the centroid of the ball.

    :param res_w: Native resolution width.
    :param res_h: Native resolution height.
    :param crop_w: Width of the ROI within the native resolution.
    :param crop_h: Height of the ROI within the native resolution.
    :param centroid_x: X-coordinate of the centroid of the ball.
    :param centroid_y: Y-coordinate of the centroid of the ball.
    :return: Tuple (offset_x, offset_y, centroid_x, centroid_y)
    """

    if centroid_x is None:
        centroid_x = crop_w // 2
    if centroid_y is None:
        centroid_y = crop_h // 2

    center_x = res_w // 2
    center_y = res_h // 2

    max_offset_x = min(center_x, res_w - crop_w)
    max_offset_y = min(center_y, res_h - crop_h)

    offset_x = max(0, min(center_x - centroid_x, max_offset_x))
    offset_y = max(0, min(center_y - centroid_y, max_offset_y))

    return offset_x, offset_y, centroid_x, centroid_y


def set_v4l2_format(width, height, centroid_x=None, centroid_y=None):
    """
    Set V4L2 format for a specified width and height.

    :param width: Desired width.
    :param height: Desired height.
    :param centroid_x: X-coordinate of the centroid (optional).
    :param centroid_y: Y-coordinate of the centroid (optional).
    :return: Path of the configured device or None.
    """
    offset_x, offset_y, centroid_x, centroid_y = calculate_offset(crop_w=width, crop_h=height, centroid_x=centroid_x, centroid_y=centroid_y)
    
    for m in range(0, 6):
        try:
            command = f"media-ctl -d /dev/media{m} --set-v4l2 \"'imx296 10-001a':0 [fmt:SBGGR10_1X10/{width}x{height} crop:({offset_x},{offset_y})/{width}x{height}]\""
            logging.info(f"Executing command: {command}")
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logging.info(f"Success configuring media-ctl on /dev/media{m}")
            return f'/dev/media{m}'
        except subprocess.CalledProcessError as e:
            logging.error(f"Error configuring media-ctl: {e}")

    return None


def list_cameras():
    """
    List available cameras using 'libcamera-hello --list-cameras'.
    """
    try:
        command = "libcamera-hello --list-cameras"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error calling 'libcamera-hello --list-cameras': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set V4L2 format and list cameras")
    parser.add_argument("--width", type=int, default=1152, help="Desired width")
    parser.add_argument("--height", type=int, default=192, help="Desired height")

    args = parser.parse_args()

    successful_device = set_v4l2_format(args.width, args.height)
    if successful_device:
        logging.info(f"Successful configuration on device: {successful_device}")
    else:
        logging.error("No successful configuration")
    
    list_cameras()
