import time
import vgamepad as vg

gamepad = vg.VX360Gamepad()

print("Throttle for 2 seconds...")
gamepad.right_trigger(value=255)
gamepad.update()
time.sleep(2)

print("Release...")
gamepad.right_trigger(value=0)
gamepad.update()
time.sleep(1)

print("Brake for 2 seconds...")
gamepad.left_trigger(value=255)
gamepad.update()
time.sleep(2)

print("Release...")
gamepad.left_trigger(value=0)
gamepad.update()

print("Done")