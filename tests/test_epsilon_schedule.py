from train import epsilon_schedule

assert abs(epsilon_schedule(0) - 1.0) < 1e-9
assert abs(epsilon_schedule(10000) - 0.525) < 1e-9
assert abs(epsilon_schedule(20000) - 0.05 < 1e-9)
assert abs(epsilon_schedule(300000) - 0.05) < 1e-9

print("Epsilon Schedule test PASSED!")
