import minari

dataset = minari.load_dataset("atari/breakout/expert-v0", download=True)


print(len(dataset[0]))
print(dataset[0].observations[4].shape[0])
