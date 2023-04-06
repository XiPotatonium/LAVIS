
import json
from datasets import load_dataset
from rich.progress import Progress, BarColumn, TimeElapsedColumn, SpinnerColumn, TextColumn, track
from rich.console import Console
from pathlib import Path
import shutil
import time
import typer
import pandas as pd


app = typer.Typer()


@app.command()
def stat_threshold(
    path: str,
):
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}: {task.completed}"),
        TimeElapsedColumn(),
        console=console,
        # get_time=lambda: 1.0,
        # auto_refresh=False,
    )
    tid = progress.add_task("Loading dataset", total=None)
    ds = load_dataset(path, split="train", streaming=True)
    quality = [[k, 0] for k in [0.0, 0.2, 0.27, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35]]
    # [[0.2, 143104279], [0.25, 143090313], [0.28, 62163103], [0.3, 22466752], [0.32, 6984718], [0.33, 3706167], [0.35, 953541]]
    with progress:
        for sample in ds:
            for i, q in enumerate(quality):
                if sample["similarity"] > q[0]:
                    quality[i][1] += 1
            progress.advance(tid)
    print(quality)


@app.command()
def get_urls(
    path: str,
    output: str = typer.Option(...),
    threshold: float = 0.0,
):
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}: {task.completed}"),
        TimeElapsedColumn(),
        console=console,
        # get_time=lambda: 1.0,
        # auto_refresh=False,
    )
    tid = progress.add_task("Loading dataset", total=None)
    ds = load_dataset(path, split="train", streaming=True)
    # quality = [[k, 0] for k in [0.2, 0.25, 0.28, 0.3, 0.32, 0.33, 0.35]]
    # [[0.2, 143104279], [0.25, 143090313], [0.28, 62163103], [0.3, 22466752], [0.32, 6984718], [0.33, 3706167], [0.35, 953541]]
    # [[0.0, 143104279], [0.2, 143104279], [0.27, 96443389], [0.3, 22466752], [0.31, 12741743], [0.32, 6984718], [0.33, 3706167], [0.34, 1898692], [0.35, 953541]]
    data = []
    with progress:
        for sample in ds:
            if sample["similarity"] > threshold:
                data.append([sample["SAMPLE_ID"], sample["URL"], sample["TEXT"], sample["similarity"]])
            progress.advance(tid)
    df = pd.DataFrame(columns=["SAMPLE_ID", "URL", "TEXT", "similarity"], data=data)
    print("Total samples:", len(df))
    Path(output).parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(output, index=False)


@app.command()
def get_urls_dev(
    path: str,
    output: str = typer.Option(...),
    threshold: float = 0.0,
    count: int = 10000,
):
    import heapq

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}: {task.completed}"),
        TimeElapsedColumn(),
        console=console,
        # get_time=lambda: 1.0,
        # auto_refresh=False,
    )
    tid = progress.add_task("Loading dataset", total=None)
    ds = load_dataset(path, split="train", streaming=True)
    # quality = [[k, 0] for k in [0.2, 0.25, 0.28, 0.3, 0.32, 0.33, 0.35]]
    # [[0.2, 143104279], [0.25, 143090313], [0.28, 62163103], [0.3, 22466752], [0.32, 6984718], [0.33, 3706167], [0.35, 953541]]
    # [[0.0, 143104279], [0.2, 143104279], [0.27, 96443389], [0.3, 22466752], [0.31, 12741743], [0.32, 6984718], [0.33, 3706167], [0.34, 1898692], [0.35, 953541]]
    data = []
    with progress:
        def ds_iter():
            for sample in ds:
                if sample["similarity"] < threshold:
                    yield sample
                progress.advance(tid)
        data = heapq.nlargest(count, ds_iter(), key=lambda x: x["similarity"])
    data = [[sample["SAMPLE_ID"], sample["URL"], sample["TEXT"], sample["similarity"]] for sample in data]
    df = pd.DataFrame(columns=["SAMPLE_ID", "URL", "TEXT", "similarity"], data=data)
    print("Total samples:", len(df))
    Path(output).parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(output, index=False)


@app.command()
def download(
    urls: str,
):
    from img2dataset import download

    output_folder = str(Path(urls).parent)
    download(
        processes_count=8,
        thread_count=16,
        url_list=urls,
        # image_size=256,
        resize_mode="no",
        output_folder=output_folder,
        output_format="webdataset",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        # enable_wandb=True,
        number_sample_per_shard=100000,
        distributor="multiprocessing",
    )

    stat(output_folder)


@app.command()
def stat(output: str):
    successes = 0
    total = 0
    failed_to_download = 0
    failed_to_resize = 0
    for f in Path(output).glob("*stats.json"):
        stats = json.loads(f.read_text())
        successes += stats["successes"]
        total += stats["count"]
        failed_to_download += stats["failed_to_download"]
        failed_to_resize += stats["failed_to_resize"]
    print(f"Total: {total}, Successes: {successes}, Failed to download: {failed_to_download}, Failed to resize: {failed_to_resize}")

    df = pd.read_parquet(output + "/urls.parquet")
    # get the laximum length
    max_len = 0
    for i, row in df.iterrows():
        max_len = max(max_len, len(row["TEXT"]))
    print("Max length:", max_len)


if __name__ == "__main__":
    app()