from pathlib import Path


def write_submission_csv(predictions, names, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)
    enumerated_names = [(i, name) for i, name in enumerate(names)]
    enumerated_names.sort(key=lambda name: name[1])

    with open(filepath, 'w') as out_file:
        out_file.write('image,level\n')
        for i, name in enumerated_names:
            prediction = predictions[i]
            out_file.write(f'{name},{prediction}\n')
        # Add labels for corrupt images using prediction for their left image.
        if "25313_left" in names:
            out_file.write(f"25313_right,{predictions[names.index('25313_left')]}\n")
        if "27096_left" in names:
            out_file.write(f"27096_right,{predictions[names.index('27096_left')]}\n")
