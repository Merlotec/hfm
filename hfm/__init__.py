from .config import HFMConfig
from .model import HFM
from .quadtree import QuadtreeHFM
from .context_encoder import ContextEncoder
from .discriminator import HFMDiscriminator
from .trainer import GANTrainer, train_step_gan


def build_model(cfg: HFMConfig):
    """Construct the model selected by `cfg.use_quadtree`."""
    return QuadtreeHFM(cfg) if cfg.use_quadtree else HFM(cfg)


__all__ = [
    "HFMConfig", "HFM", "QuadtreeHFM", "build_model", "ContextEncoder",
    "HFMDiscriminator", "GANTrainer", "train_step_gan",
]
