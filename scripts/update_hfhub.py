import json
import typer
import shutil
import os
from pathlib import Path


app = typer.Typer()


@app.command()
def main(src: str, target: str = typer.Option(...)):
    src = Path(src)
    target = Path(target)
    shutil.copy(src / "pytorch_model.bin", target / "pytorch_model.bin")
    with (src / "config.json").open("r", encoding="utf8") as rf:
        config = json.load(rf)
    config["auto_map"] = {
        "AutoConfig": "configuration_blip2chatglm.Blip2ChatGLMConfig",
        "AutoModel": "modeling_blip2chatglm.Blip2ForChatGLM",
        "AutoModelForCausalLM": "modeling_blip2chatglm.Blip2ChatGLM",
    }
    with (target / "config.json").open("w", encoding="utf8") as wf:
        json.dump(config, wf, indent=2, ensure_ascii=False)
    # shutil.copy(src / "config.json", target / "config.json")
    shutil.copy(
        "lavis/models/blip2zh_models/blip2zh_chatglm/configuration_blip2chatglm.py",
        target / "configuration_blip2chatglm.py",
    )
    shutil.copy(
        "lavis/models/blip2zh_models/blip2zh_chatglm/modeling_blip2chatglm.py",
        target / "modeling_blip2chatglm.py",
    )
    shutil.copy(
        "lavis/models/blip2zh_models/blip2zh_chatglm/configuration_chatglm.py",
        target / "configuration_chatglm.py",
    )
    shutil.copy(
        "lavis/models/blip2zh_models/blip2zh_chatglm/modeling_chatglm.py",
        target / "modeling_chatglm.py",
    )


if __name__ == "__main__":
    app()
