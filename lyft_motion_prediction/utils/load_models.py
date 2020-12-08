from typing import Dict
from collections import namedtuple
from torch.utils import model_zoo
from lyft_motion_prediction.models import CVAE, TrajectoriesExtractor


model = namedtuple("model", ["url", "model"])

models = {
    "CVAE": model(
        url="https://github.com/kumgleb/Lyft_motion_prediction/releases/download/weights/CVAE_weights.zip",
        model=CVAE,
    ),
    "Extractor": model(
        url="https://github.com/kumgleb/Lyft_motion_prediction/releases/download/weights/TrajectoriesExtractor_weights",
        model=TrajectoriesExtractor,
    )
}


def get_model(model_name: str, cfg: Dict, device: str):
    model = models[model_name].model(cfg).to(device)
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location=device)

    model.load_state_dict(state_dict)

    return model

