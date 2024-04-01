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


    def list_configurations(self):
        print('Index | Config name\n--------------------------')
        for config in self.configs:
            print(str(list(self.configs).index(config)).ljust(5, ' ') + '   ' + config)

    def print_config(self, config_name):
        if config_name in self.configs:
            print(self.configs[config_name])
        else:
            print(list(self.configs.items())[int(config_name)][1])


    def run(self, config_name):
        c, config_name = select(self.configs, config_name)

        ts = datetime.now().strftime('%d-%m-%Y_%H-%M')
        instance_name = input(f'Instance name [{config_name}_{ts}]: ')
        if instance_name == '':
            instance_name = f'{config_name}_{ts}'

        print(f'Running "{instance_name}" ({config_name})...')
        r = Runner(instance_name, self.configs[config_name])
        r.run()
        self.running[instance_name] = r
    
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
    