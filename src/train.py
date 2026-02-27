import hydra
import os
import json
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig 
from config import InditexConfig

cs = ConfigStore.instance()
cs.store(name="inditex_config", node=InditexConfig)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: InditexConfig):
    # 1. Get the output directory from Hydra configuration
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")

    # -----------------------------------------
    # Here goes your training logic
    # -----------------------------------------
    print(cfg.params)
    resultados = {"loss": 0.23, "accuracy": 0.92} # Simulated results

    # 2. Save metrics/results to a JSON file
    resultados_path = os.path.join(output_dir, "resultados.json")
    with open(resultados_path, "w") as f:
        json.dump(resultados, f, indent=4)
    print(f"Resultados guardados en: {resultados_path}")

    # 3. Save model weights (simulated here, replace with actual model saving code)
    pesos_path = os.path.join(output_dir, "model_weights.pth")
    # torch.save(modelo.state_dict(), pesos_path)

if __name__ == "__main__":
    main()