import subprocess
import argparse

def set_v4l2_format(width, height):
    print(f"width: {width} height: {height}\n")
    # Loop through media devices and set V4L2 format
    for m in range(1, 6):
        try:
                # Calculate crop values
            crop_x = (1440 - width) // 2
            crop_y = (1088 - height) // 2

            # Construct the command using triple-quotes to handle both single and double quotes
            command = f'''media-ctl -d /dev/media{m} --set-v4l2 "'imx296 10-001a':0 [fmt:SBGGR10_1X10/{width}x{height} crop:({crop_x},{crop_y})/{width}x{height}]" '''
 
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Success configuring media-ctl: on /dev/media{m}\n")
            break
        except subprocess.CalledProcessError as e:
            print(f"Error configuring media-ctl: {e}")

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

    set_v4l2_format(args.width, args.height)
    list_cameras()
