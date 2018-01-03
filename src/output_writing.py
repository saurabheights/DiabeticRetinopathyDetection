from pathlib import Path


def write_submission_csv(predictions, names, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)
    names = [(i, name) for i, name in enumerate(names)]
    names.sort(key=lambda name: name[1])

    with open(filepath, 'w') as out_file:
        out_file.write('image,level\n')
        for i, name in names:
            prediction = predictions[i]
            out_file.write(f'{name},{prediction}\n')
