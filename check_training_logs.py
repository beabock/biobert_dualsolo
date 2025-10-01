import json, glob, os

checkpoints = glob.glob('results_classification/checkpoint-*/trainer_state.json')
if checkpoints:
    latest = sorted(checkpoints)[-1]
    print(f'Checking latest checkpoint: {latest}')
    with open(latest, 'r') as f:
        data = json.load(f)
    log_history = data.get('log_history', [])
    print(f'Log history entries: {len(log_history)}')
    if log_history:
        print('Sample entries:')
        for i, entry in enumerate(log_history[:5]):
            step = entry.get('step', 0)
            loss = entry.get('loss', 'N/A')
            eval_loss = entry.get('eval_loss', 'N/A')
            eval_acc = entry.get('eval_accuracy', 'N/A')
            print(f'  Entry {i}: step={step}, loss={loss}, eval_loss={eval_loss}, eval_acc={eval_acc}')
    else:
        print('No log history found')
else:
    print('No checkpoints found')