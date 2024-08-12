from copy import deepcopy

from good_rainbow_src.utils import merge_dicts
from runner_src.runner import *
from datetime import datetime


def select(dictionary, key_or_index):
    try:
        return dictionary[key_or_index], key_or_index
    except KeyError:
        return list(dictionary.items())[int(key_or_index)][1], list(dictionary.keys())[int(key_or_index)]

class Core():
    def __init__(self, configs):
        self.configs = configs
        self.running = {}

    def menu(self):
        while(True):
            print('> ', end='')
            cmd = input().split()

            if cmd[0] in ['lc', 'list_configs']:
                self.list_configurations()

            elif cmd[0] in ['r', 'run']:
                self.run(cmd[1])

            elif cmd[0] in ['pc', 'print_cfg']:
                pass
                self.print_config(cmd[1])

            elif cmd[0] in ['li', 'list_instances']:
                self.list_running()

            elif cmd[0] in ['s', 'save']:
                self.save_state(cmd[1])

            elif cmd[0] in ['stop']:
                self.stop(cmd[1])

            elif cmd[0] in ['quit']:
                if self.running:
                    print('Ignoring quit: some instance(s) still running')
                else:
                    break

            else:
                print(
                    '=============================\n'
                    'Available commands:\n'
                    '> lc, list_cfgs\n'
                    '> r, run_cfg <config-name / config-index>\n'
                    '> pc, print_cfg <config-name / config-index>\n'
                    '> li, list_instances\n'
                    '> s, save <instance-name / instance-index>\n'
                    '> stop <instance / instance-index>\n'
                    '> quit\n'
                    '=============================\n'
                )

    def run_from_console(self, experiment_plan):
        for config_id, additional_settings in experiment_plan:
            print("running")
            c, config_name = select(self.configs, config_id)
            print(config_name)
            variant = additional_settings["meta"]["variant"]
            if "APDPR" in variant or variant == "PRQL" or variant == "DIRECT_WEIGHT_REUSE":
                if "MiniGrid" in config_name:
                    files = list(sorted(os.listdir("sorted_models/minigrid")))
                    print(len(files))
                    files = [i for i in files if "-".join(config_name.split('_')[0:1]) not in i]
                    if variant == "DIRECT_WEIGHT_REUSE":
                        files_similar = [i for i in files if "-".join(config_name.split("-")[0:2]) in i]
                        print("dupa")
                        print(files_similar)
                        print("-".join(config_name.split("-")[0:2]))
                        if len(files_similar) == 0:
                            files = list(sorted(os.listdir("sorted_models/minigrid")))
                            indexes = [i for i, j in enumerate(files) if "-".join(config_name.split('_')[0:1]) in j]
                            if len(indexes) == 0:
                                index = 0
                            else:
                                if indexes[0] - 1 >=0:
                                    index = indexes[0] - 1
                                else:
                                    index = indexes[0] + 1

                            files = [files[index]]
                        else:
                            files = [files_similar[0]]

                    print(len(files))
                    additional_settings["meta"]["guided_network"] = ["sorted_models/minigrid/" + i for i in  files]
                elif "FrozenLake" in config_name:
                    files = list(sorted(os.listdir("sorted_models/frozen_lake")))
                    print(len(files))
                    files = [i for i in files if "_".join(config_name.split('_')[0:2]) not in i]
                    if variant == "DIRECT_WEIGHT_REUSE":
                        print(files)
                        files = [files[0]]
                    print(len(files))
                    additional_settings["meta"] ["guided_network"] =  ["sorted_models/frozen_lake/" + i for i in files]
                elif "lunar_lander" in config_name:
                    files = list(sorted(os.listdir("sorted_models/lunar_lander")))
                    print(len(files))
                    files = [i for i in files if "_".join(config_name.split('_')[0:5]) not in i]
                    if variant == "DIRECT_WEIGHT_REUSE":
                        print(files)
                        files = [files[0]]
                    print(len(files))
                    additional_settings["meta"]["guided_network"] = ["sorted_models/lunar_lander/" + i for i in files]

            self.run(config_id, additional_settings, console_run=True)

    def list_configurations(self):
        print('Index | Config name\n--------------------------')
        for config in self.configs:
            print(str(list(self.configs).index(config)).ljust(5, ' ') + '   ' + config)

    def print_config(self, config_name):
        if config_name in self.configs:
            print(self.configs[config_name])
        else:
            print(list(self.configs.items())[int(config_name)][1])


    def run(self, config_name, additional_settings=None, console_run=False):
        c, config_name = select(self.configs, config_name)

        ts = datetime.now().strftime('%d-%m-%Y_%H-%M')
        default_name = f'{config_name}_{ts}'
        if not console_run:
            instance_name = input(f'Instance name [{config_name}_{ts}]: ')
        else:
            instance_name = default_name

        if instance_name == '':
            instance_name = default_name

        print(f'Running "{instance_name}" ({config_name})...')

        if additional_settings is None:
            additional_settings = {}

        config_to_run = deepcopy(self.configs[config_name])

        config_to_run = merge_dicts(config_to_run, additional_settings)
        r = Runner(instance_name, config_to_run)
        r.run(console_run=console_run)
        r = None
        #self.running[instance_name] = r
    
    def list_running(self):
        print('Index | Instance name\n--------------------------')
        for instance in self.running:
            print(str(list(self.running).index(instance)).ljust(5, ' ') + '   ' + instance)
    
    def save_state(self, instance_name):
        instance, instance_name = select(self.running, instance_name)
        instance.save()

    def stop(self, instance_name):
        instance, instance_name = select(self.running, instance_name)
        instance.stop()
        instance.save()
        self.running.pop(instance_name)
    