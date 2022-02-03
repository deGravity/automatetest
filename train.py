from automate import run_model

if __name__ == '__main__':
    run_model(default_args = {
        'tensorboard_path': '/projects/grail/benjones/logs/hsearchtests',
        'name': 'hptest',
        'seed':42,
        'model_class':'automate.FixedGridPredictor',
        'debug':True
    })

