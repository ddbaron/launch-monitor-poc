from sense_hat import SenseHat
import argparse
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Initialize the Sense HAT object
    sense = SenseHat()
except Exception as e:
    logging.error(f"Failed to initialize Sense HAT: {e}")
    exit(1)

def set_leds(color):
    """
    Set the LEDs to the specified color.

    :param color: A string (color name) or a tuple (R, G, B) to set the color of the LEDs.
    """
    # Define color dictionary
    color_dict = {
        'off': (0, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'red': (255, 0, 0)
    }

    # Check if the color is a string and get the corresponding tuple
    if isinstance(color, str):
        color = color_dict.get(color.lower(), (0, 0, 0))  # Default to 'off' if color not found

    try:
        sense.clear(color)  # Set color
    except Exception as e:
        logging.error(f"Error setting LEDs: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control the Sense HAT LED panel")
    parser.add_argument("--color", type=str, default='off', help="Color of the LEDs (name or RGB tuple)")

    args = parser.parse_args()

    try:
        color_input = args.color
        # Check if the input is tuple-like and convert it
        if ',' in color_input:
            color_input = tuple(map(int, color_input.split(',')))

        set_leds(color_input)
    except Exception as e:
        logging.error(f"Failed to execute set_leds function: {e}")
