

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_path",
    required=True,
    help="the path of the standalone experiment.",
)
parser.add_argument(
    "--task_output",
    help="the output task schema.",
    required=False,
    default='task',
)
args = parser.parse_args()

    

for test in range(10):
    for sort in range(9):

        train_path = args.experiment_path + f'/cnn_fold{test}/sort{sort}'

        weights_path = os.path.join(train_path, "model_weights.pkl")
        history_path = os.path.join(train_path, "history.pkl")
        params_path  = os.path.join(train_path, "parameters.pkl")

        with open(history_path, 'r') as f:
            history = json.load(f)
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        with open(weights_path, 'r') as f:
            weights = json.load(f)

        image_shape = model_params["image_shape"]
        model = create_cnn(image_shape)
        model.set_weights(weights)

        # build the original model
        train_state = prepare_model(model, history, model_params)



