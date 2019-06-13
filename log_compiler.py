def compile(numepoch, batchsize, lrate, accuracy, precision, f1, tp, tn, fp, fn, fc, training_number, model):
    total = tn + tp + fp + fn
    fc = '\n'.join(list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(map(lambda x: x[1], fc.named_children()), 1))))
    return str(f'# Training number: {training_number} with model: {model}\n'
               f'## Hyper Parameters:\n'
               f'- Epoch numbers: {numepoch}\n'
               f'- Batch size: {batchsize}\n'
               f'- Learning rate: {lrate}\n'
               f'\n'
               f'## Scores\n'
               f'- Accuracy: {accuracy:.2f}\n'
               f'- Precision: {precision:.2f}\n'
               f'- F1: {f1:.2f}\n'
               f'\n'
               f'## Confusion Matrix\n'
               f'- Predicted true and actually true: {tp}\n'
               f'- Predicted false and actually false: {tn}\n'
               f'- Predicted true but actually false: {fp}\n'
               f'- Predicted false but actually true: {fn}\n'
               f'- Total correct predictions: {tp + tn} ({(tp + tn) / total * 100:.2f}%)\n'
               f'- Total wrong predictions: {fn + fp} ({(fn + fp) / total * 100:.2f}%)\n'
               f'- Total: {total}\n'
               f'\n'
               f'## Fully connected layer:\n'
               f'{fc}')


def write_log_file(filename, data):
    with open(f'logs/{filename}.md', 'w') as f:
        f.write(data)
