from sense_hat import SenseHat
import time
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

def set_leds(color, flash=False):
    """
    Set the LEDs to the specified color. Optionally, flash the LEDs.

    :param color: A tuple (R, G, B) to set the color of the LEDs.
    :param flash: Boolean to indicate if the LEDs should flash.
    """
    try:
        if flash:
            for _ in range(3):  # Flash 3 times
                sense.clear(color)  # Set color
                time.sleep(0.5)     # On for 0.5 seconds
                sense.clear()       # Turn off
                time.sleep(0.5)     # Off for 0.5 seconds
        else:
            sense.clear(color)  # Set color without flashing
    except Exception as e:
        logging.error(f"Error setting LEDs: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control the Sense HAT LED panel")
    parser.add_argument("--color", type=str, choices=['off', 'blue', 'green', 'yellow', 'red'], default='off', help="Color of the LEDs")
    parser.add_argument("--flash", action='store_true', help="Enable flashing LEDs")

    args = parser.parse_args()

    colors = {
        'off': (0, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'red': (255, 0, 0)
    }

    try:
        set_leds(colors[args.color], args.flash)
    except Exception as e:
        logging.error(f"Failed to execute set_leds function: {e}")
