import json
from pathlib import Path
from math import ceil
import csv
import gc

INPUT_DIR = Path("raw_data")
OUTPUT_DIR = Path("data")

def remove(location: Path):
    for subloc in location.iterdir():
        if subloc.is_dir() and subloc.name == "images":  # I am weak enough to fear "rm -rf"
            remove(subloc)
        else:
            subloc.unlink()


def load_file(filepath: Path) -> list[dict]:
    with open(str(filepath)) as fp:
        text_file = fp.readlines()
    return [json.loads(line) for line in text_file]


def preprocess_data(json_data: list[dict]) -> dict:
    return {
        "word": json_data[0]["word"],
        "images": [
            image["drawing"] for image in json_data
            if image["recognized"]
        ]
    }


def draw_line(src: list[list[int]], x0: int, y0: int, x1: int, y1: int):
    a_param = (y0 - y1) / (x0 - x1)
    b_param = y0 - a_param * x0

    start = min([x0, x1])
    stop = max([x0, x1])

    for x_step in range(start, stop + 1):
        raw_ypred = a_param * x_step + b_param
        potential_targets = [
            ceil(raw_ypred),
            int(raw_ypred),
            ceil(raw_ypred + 0.5),
            ceil(raw_ypred - 0.5),
            int(raw_ypred + 0.5),
            int(raw_ypred - 0.5)
        ]
        for target in potential_targets:
            try:
                src[target][x_step] = 1
            except IndexError:
                pass


def draw_hline(src, x, y0, y1):
    for y_step in range(min([y0, y1]), max([y0, y1])):
        src[y_step][x] = 1


def strokes_to_image(drawing: list) -> list:
    result = [[0 for _ in range(256)] for _ in range(256)]

    for line in drawing:
        xs = zip(line[0], line[0][1:])
        ys = zip(line[1], line[1][1:])

        for ((x0, x1), (y0, y1)) in zip(xs, ys): 
            result[y0][x0] = 1
            result[y1][x1] = 1

            if x1 != x0:
                draw_line(result, x0, y0, x1, y1) 
            else:
                draw_hline(result, x0, y0, y1)
    return result


def flatten_arr(arr: list[list[int]]) -> list[int]:
    result = [] 
    for row in arr:
        result.extend(row)
    return result


def save_to_ppm(name: str, image: list[list[int]]):
    result = "P1\n256 256\n"
    for row in image:
        result += ''.join([str(val) for val in row]) + '\n'

    out_subdir = OUTPUT_DIR / "images" 
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_file = out_subdir / (name + ".ppm")
    
    with open(out_file, "w+") as fp:
        fp.write(result)


def save_to_csv(name: str, images: list[list[int]]):
    out_file = OUTPUT_DIR / (name + '.csv')

    with open(out_file, 'w+', newline='') as fp:
        wirter = csv.writer(fp)
        wirter.writerow(["word", *[f"pixel[{i}]" for i in range(256 * 256)]])
        for image in images:
            wirter.writerow([name, *image])

def preprocess_all(batch_size: int):
    for file in INPUT_DIR.iterdir():
        print("=" * 10)
        print("Loading file: ", str(file))
        data = preprocess_data(load_file(file))
        category_images = []
        gc.collect()
        size = min([len(data["images"]), batch_size])
        print(f"Batch size: {size}")
        for i, image in enumerate(data["images"][:size]):
            image = strokes_to_image(image)
            save_to_ppm(data["word"] + str(i), image)
            category_images.append(flatten_arr(image))
        save_to_csv(data["word"], category_images)
        print(f"Finished processing file: {str(file)}")


if __name__ == '__main__':
    print("Startig")
    print(f"Cleaning: {str(OUTPUT_DIR)}")
    remove(OUTPUT_DIR)
    category_size = 10_000
    print("Starting preprocessing")
    preprocess_all(category_size)
