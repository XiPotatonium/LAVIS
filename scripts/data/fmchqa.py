import typer
from pathlib import Path
import gzip
import tarfile
import json
from loguru import logger
from . import dir_iter


app = typer.Typer()


@app.command()
def main(path: str, imgpath: str = typer.Option(...)):
    with tarfile.open(path, 'r') as file:
        data = file.extractfile("FM-CH-QA.json")
        data = json.load(data)

    img_map = {}
    for img in dir_iter(Path(imgpath)):
        # strip leading 0s
        img = Path(img)
        img_map[str(int(img.stem.split('_')[-1]))] = img.name

    samples = []
    for sample in data["train"]:
        try:
            image = img_map[sample["image_id"]]
        except KeyError as e:
            logger.error(f"Image {sample['image_id']} not found")
            continue
        samples.append({
            "image": image,
            "question": sample["Question"],
            "answer": sample["Answer"],
        })
    logger.info(f"Get {len(samples)} samples")
    with (Path(path).parent / "train.json").open('w', encoding="utf8") as wf:
        json.dump(samples, wf, ensure_ascii=False)


if __name__ == "__main__":
    app()
