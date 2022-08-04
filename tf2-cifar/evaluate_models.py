import argparse
import utils
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorFlow2.0 CIFAR-10 Training')
    parser.add_argument('checkpoint_paths', nargs='+', type=str, help="path to the model's checkpoint")
    parser.add_argument('--model', default='resnet18', type=str, help='model type')
    args = parser.parse_args()
    args.model = args.model.lower()
    train_ds, test_ds, decay_steps = utils.prepare_data(1.0, 256, 1)

    print('==> Building model...')
    model = utils.create_model(args.model, num_classes=10)
    for checkpoint_path in args.checkpoint_paths:
        print(checkpoint_path)
        utils.load_weights_to_model(model, checkpoint_path)
        accuracy = utils.evaluate_model(model, test_ds=test_ds)
        utils.write_json_to_file(data={"Test Accuracy": float(accuracy)}, filename=os.path.join(checkpoint_path, "log.json"))
