import numpy as np
from byte_triggers import ParallelPortTrigger
from psyquartz import Clock, sleepers

clock = Clock()
parport = "/dev/parport0"

durations = np.arange(50, 1, -1)
trigvals = np.repeat(np.arange(1, 17), 5)
long_dur = 0.5
short_dur = 0.1

print("Sending 16 triggers of 1s duration, 100ms apart.")
trigger = ParallelPortTrigger(parport, delay=1000)
for trig in np.arange(1, 17):
    trigger.signal(trig)
    sleepers(1.1)
trigger.close()
del trigger

for dur in durations:
    print(
        f"Testing {dur} ms trigger duration. 80 triggers {short_dur*1000:0.0f}ms apart, 16 triggers {long_dur*1000:0.0f}ms apart. "
        f"Should take {80*short_dur + 16*long_dur:0.2f}s."
    )
    trigger = ParallelPortTrigger(parport, delay=dur)
    trigger.signal(255)
    sleepers(short_dur)
    for trig in trigvals:
        trigger.signal(trig)
        sleepers(short_dur)
    trigger.signal(255)
    sleepers(short_dur)
    for trig in np.arange(1, 17):
        trigger.signal(trig)
        sleepers(long_dur)
    trigger.close()
    del trigger
