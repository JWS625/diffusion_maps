from pathlib import Path
from diffusion_maps.src.dm_main import DMClass
from diffusion_maps.src.krr_model import Modeler


root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data"
model_dir = root_dir / "trained_mdls"

