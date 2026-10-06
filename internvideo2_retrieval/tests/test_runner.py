from pathlib import Path
from run_datasets import make_parser, commands_for


def test_runner_selects_dataset_specific_outputs_and_cuda():
    args = make_parser().parse_args(['--base-root', '/data', '--output-root', '/features'])
    commands = commands_for(args, 'vatex')
    assert '--dataset' in commands['extract']
    assert commands['extract'][commands['extract'].index('--dataset')+1] == 'vatex'
    assert commands['extract'][commands['extract'].index('--data-root')+1] == str(Path('/data/vatex'))
    assert commands['extract'][commands['extract'].index('--device')+1] == 'cuda'
    assert commands['evaluate'][commands['evaluate'].index('--features-dir')+1] == str(Path('/features/vatex'))
    assert '--check-data' in commands['check']


def test_runner_supports_user_root_and_segment_mode():
    args = make_parser().parse_args(['--msrvtt-root', '/custom', '--activitynet-mode', 'video'])
    command = commands_for(args, 'msrvtt')['extract']
    assert command[command.index('--data-root')+1] == '/custom'
    command = commands_for(args, 'activitynet_captions')['extract']
    assert command[command.index('--activitynet-mode')+1] == 'video'
