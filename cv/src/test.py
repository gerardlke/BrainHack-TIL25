from pathlib import Path, PurePath

print(len(sorted(Path('/home/jupyter/BrainHack-TIL25/cvdataset/images/train').rglob('*jpg'))))
print(len(sorted(Path('/home/jupyter/BrainHack-TIL25/cvdataset/labels/train').rglob('*txt'))))

print(len(sorted(Path('/home/jupyter/BrainHack-TIL25/cvaugmented/images/train').rglob('*jpg'))))
print(len(sorted(Path('/home/jupyter/BrainHack-TIL25/cvaugmented/labels/train').rglob('*txt'))))

print(len(sorted(Path('/home/jupyter/BrainHack-TIL25/cvdataset/images/val').rglob('*jpg'))))
print(len(sorted(Path('/home/jupyter/BrainHack-TIL25/cvdataset/labels/val').rglob('*txt'))))

print(len(sorted(Path('/home/jupyter/BrainHack-TIL25/cvaugmented/images/val').rglob('*jpg'))))
print(len(sorted(Path('/home/jupyter/BrainHack-TIL25/cvaugmented/labels/val').rglob('*txt'))))