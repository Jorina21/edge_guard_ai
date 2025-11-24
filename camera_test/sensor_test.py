import RPi.GPIO as GPIO, time

PIR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

print("Watching PIR output...")
prev = GPIO.input(PIR_PIN)
try:
    while True:
        curr = GPIO.input(PIR_PIN)
        if curr != prev:
            print("Changed to", "HIGH" if curr else "LOW", "at", time.strftime("%H:%M:%S"))
            prev = curr
        time.sleep(0.1)
except KeyboardInterrupt:
    GPIO.cleanup()

