from pathlib import Path
from diffusion_maps.src.dm_main import DMClass
from diffusion_maps.src.krr_model import Modeler


root_dir = Path(__file__).resolve().parent
data_dir = str(root_dir / "data")
model_dir = str(root_dir / "trained_mdls")
external_dir = str(root_dir / "external")

