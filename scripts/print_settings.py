from mi.experiments.settings import list_settings
from mi.experiments.settings.data_models import Setting
from mi import config
from pathlib import Path

def describe_setting(setting: Setting):
    print(f"Setting: {setting.get_domain_name()}")
    print(f"Control dataset path: {Path(setting.get_control_dataset_path()).relative_to(config.DATASETS_DIR)}")
    print(f"Finetuning dataset path: {Path(setting.get_finetuning_dataset_path()).relative_to(config.DATASETS_DIR)}")
    try: 
        print(f"Task specific inoculation: {setting.get_task_specific_inoculation()}")
    except NotImplementedError:
        print("Task specific inoculation: Not implemented")
    try:
        print(f"Control inoculation: {setting.get_control_inoculation()}")
    except NotImplementedError:
        print("Control inoculation: Not implemented")
    try:
        print(f"General inoculation: {setting.get_general_inoculation()}")
    except NotImplementedError:
        print("General inoculation: Not implemented")
    print()

for setting in list_settings():
    describe_setting(setting)