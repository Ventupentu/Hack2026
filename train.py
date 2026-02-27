# In order to have autocompletion for the config, paste in the shell:
# eval "$(python train.py --shell-completion install=bash)"

import hydra
from hydra.core.config_store import ConfigStore
from config import InditexConfig

cs = ConfigStore.instance()
cs.store(name="inditex_config", node=InditexConfig)

@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: InditexConfig):
    print(cfg.params)
    print(cfg.files)

# Reset settings to default values
#settings.reset()
if __name__ == "__main__":
    main()