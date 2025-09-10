Notes for AI. 

1. Where you would normally use `python`, always use `pdm run python` instead. This ensures we use the correct Python interpreter (inside the virtual environment).

2. When setting up a new experiment directory, always create a `.gitignore` file that ignores the `training_data/` folder. This prevents large generated datasets from being committed to git.