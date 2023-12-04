import subprocess
import argparse


def calculate_offset(res_w=1440, res_h=1088, crop_w=1152, crop_h=192, centroid_x=None, centroid_y=None):
    # Set default centroid values to the center of the crop
    if centroid_x is None:
        centroid_x = crop_w // 2
    if centroid_y is None:
        centroid_y = crop_h // 2

    # Calculate the center of the resolution
    center_x = res_w // 2
    center_y = res_h // 2

    # Calculate the maximum allowed offset to keep the centroid within the crop
    max_offset_x = min(center_x, res_w - crop_w)
    max_offset_y = min(center_y, res_h - crop_h)

    # Calculate the desired offset to pan the centroid to the center
    offset_x = max(0, min(center_x - centroid_x, max_offset_x))
    offset_y = max(0, min(center_y - centroid_y, max_offset_y))

    return offset_x, offset_y, centroid_x, centroid_y

def set_v4l2_format(width, height, centroid_x=None, centroid_y=None):
    #print(f"Setting V4L2 format for width: {width}, height: {height}")

    offset_x, offset_y, centroid_x, centroid_y = calculate_offset(crop_w=width,crop_h=height, centroid_x=centroid_x, centroid_y=centroid_y)
    
    # Loop through media devices and set V4L2 format
    for m in range(0, 6):
        try:
            # gscrop: (144, 448)/1152x192 crop
            # mine: (576, 96)/880x192 crop
            
            command = f'media-ctl -d /dev/media{m} --set-v4l2 "\'imx296 10-001a\':0 [fmt:SBGGR10_1X10/{width}x{height} crop:({offset_x},{offset_y})/{width}x{height}]"'
            print(f"Command: {command}")
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            #print(f"Success configuring media-ctl on /dev/media{m}\n")
            return f'/dev/media{m}'
            break
        except subprocess.CalledProcessError as e:
            print(f"Error configuring media-ctl: {e}")
    return None

def list_cameras():
    # Display the results of calling 'libcamera-hello --list-cameras'
    try:
        command = "libcamera-hello --list-cameras"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error calling 'libcamera-hello --list-cameras': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set V4L2 format and list cameras")
    parser.add_argument("--width", type=int, default=1152, help="Desired width")
    parser.add_argument("--height", type=int, default=192, help="Desired height")

    args = parser.parse_args()

    successful_device = set_v4l2_format(args.width, args.height)
    if successful_device:
        print(f"Successful configuration on device: {successful_device}")
    else:
        print("No successful configuration")
        
    list_cameras()
