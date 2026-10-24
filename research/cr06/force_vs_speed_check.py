"""Does each measured dataset's force column agree with its OWN boat speed?

Yesterday's [LO00] comparison left the model's pin force 22% low and could not say
whether that was the model or the source. This settles it, by running her stroke at
the handle-force scale that makes the model settle at each dataset's measured boat
speed, and then reading the model's stretcher and gate force there.

The test is fair rather than circular because the hull drag was never fitted to
[CR06]: it is Michell's integral checked against Lazauskas's printed Wigley curve,
Sretenskii's finite-depth integral, and ITTC'57 friction. So it can legitimately
mediate between one column of a paper and another.

Runs recorded (stretcher_from_balance_cr06.py with F_SCALE, 20 strokes):
    F_SCALE 0.865 -> 3.7458 m/s    stretcher 583  gate 654
    F_SCALE 0.908 -> 3.8583 m/s    stretcher 619  gate 693
    F_SCALE 0.969 -> 4.0139 m/s    stretcher 668  gate 749
    F_SCALE 1.000 -> 4.0901 m/s    stretcher 715  gate 800   (her own stroke)
    F_SCALE 1.128 -> 4.3948 m/s    stretcher 796  gate 893

CAVEAT, stated wherever this is quoted: every run keeps HER stroke shape and HER
rate (1.94 s, 30.9 spm). [LO00]'s subjects rowed at 30, 28 and 31 spm, so that
comparison is effectively rate-matched. [LE26]'s women rowed at 34.1 spm, where the
same power is spread over a shorter stroke, so the model's gate there would fall
roughly 10% if the rate were matched too. The [LO00] comparison is the clean one.
"""
RUNS = [(0.865, 3.7458, 583, 654), (0.908, 3.8583, 619, 693),
        (0.969, 4.0139, 668, 749), (1.128, 4.3948, 796, 893)]

# source, subject, measured speed, measured stretcher (both feet), measured gate (both pins), rate
MEAS = [("[LO00]", "A", 4.01, 358 + 338, 533 + 465, 30),
        ("[LO00]", "C", 3.84, 299 + 354, 546 + 438, 31),
        ("[LO00]", "B", 3.71, 373 + 390, 600 + 503, 28),
        ("[LE26]", "women", 4.40, 864, 916, 34.1)]

print("Model run at each dataset's OWN measured boat speed")
print("  %-8s %-6s %6s %7s | %-18s | %-18s" % ("source", "subj", "v_meas", "v_model",
                                               "stretcher  model/meas", "gate       model/meas"))
for src, subj, v, st_m, gt_m, rate in MEAS:
    run = min(RUNS, key=lambda r: abs(r[1] - v))
    _, v_mod, st, gt = run
    print("  %-8s %-6s %6.2f %7.4f | %4d vs %4d  %+6.1f%% | %4d vs %4d  %+6.1f%%"
          % (src, subj, v, v_mod, st, st_m, 100 * (st / st_m - 1), gt, gt_m, 100 * (gt / gt_m - 1)))

print("\nWhat the two columns do as the boat slows ([LO00], its own three subjects):")
print("  %-6s %6s %10s %10s | %10s %10s" % ("subj", "v", "meas pin", "model pin", "meas foot", "model foot"))
for src, subj, v, st_m, gt_m, rate in sorted([m for m in MEAS if m[0] == "[LO00]"], key=lambda m: -m[2]):
    _, v_mod, st, gt = min(RUNS, key=lambda r: abs(r[1] - v))
    print("  %-6s %6.2f %10d %10d | %10d %10d" % (subj, v, gt_m, gt, st_m, st))
print("  The model's forces fall with speed, as they must. [LO00]'s pin column does not:")
print("  its SLOWEST boat carries its HIGHEST pin force. The paper says so itself --")
print("  'Subject B applied the highest propulsive forces on the pin and the stretcher")
print("  but the boat velocity was the slowest.' B is also at the lowest rate, 28 spm,")
print("  which raises per-stroke force at equal power -- but only by about 7%, not 40%.")
