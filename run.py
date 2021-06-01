import argparse
import os
import sys

import yagmail

from targets import targets_dict, from_email, to_email

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, nargs="+", required=True)
parser.add_argument('-n', '--max_trials_num', type=int, default=3)
parser.add_argument('--notify', action='store_true', default=False)
parser.add_argument('--force_run', action='store_true', default=False)
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--gpus', type=eval, default=0)
parser.add_argument('--show_commands_only', action='store_true', default=False)
args = parser.parse_args()

target = args.target
max_trials_num = args.max_trials_num
notify = args.notify
force_run = args.force_run
show_commands_only = args.show_commands_only
skip = args.skip
gpus = args.gpus


def send_notification(status, target_name, commands):
    contents = [
        '<h2>commands:</h2>',
        '\n\n'.join(commands)
    ]
    yag.send(to_email, f'Target {target_name} {status}', contents)


def send_notification_stderr(last_command, stderr_path, status, target_name):
    try:
        with open(stderr_path) as fp:
            stderr = fp.read()
    except FileNotFoundError:
        stderr = '?'
    contents = [
        '<h2>last command:</h2>',
        last_command,
        '<h2>stderr:</h2>',
        stderr[-1000:]
    ]
    yag.send(to_email, f'Target {target_name} {status}', contents)


run_targets = {target_name: targets_dict(gpus)[target_name][skip:] for target_name in target}
for target_name, commands in run_targets.items():
    print(f'###### Target {target_name} ######')
    print('\n'.join(commands))
if show_commands_only:
    sys.exit()

if notify:
    yag = yagmail.SMTP(from_email)

for target_name, commands in run_targets.items():
    if notify:
        send_notification('STARTED', target_name, commands)
    stderr_dir = os.path.join('stderr', target_name)
    os.makedirs(stderr_dir, exist_ok=True)
    for cmd_idx, cmd in enumerate(commands, 1):
        trial_no = 1
        while trial_no <= max_trials_num:
            print(f'Running command {cmd_idx} (trial {trial_no}): {cmd}')
            stderr_file = os.path.join(stderr_dir, f'cmd_{cmd_idx}_trial_{trial_no}.txt')
            ret = os.system(f"{cmd} 2> {stderr_file}")

            if ret == 2:
                print("\nAbortion. Shutting down...")
                if notify:
                    send_notification_stderr(cmd, stderr_file, 'ABORTED', target_name)
                sys.exit()
            if ret == 0:
                os.system(f'rm -f {stderr_file}')
                break
            trial_no += 1

        if trial_no > max_trials_num:
            print("\nMaximal number of trials exceeded! Shutting down...")
            if os.path.exists(stderr_file):
                os.system(f'cat {stderr_file}')
            if notify:
                send_notification_stderr(cmd, stderr_file, 'FAILED', target_name)
            if not force_run:
                break

    if notify:
        send_notification('SUCCEEDED', target_name, commands)
