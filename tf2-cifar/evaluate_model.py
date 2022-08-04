import argparse
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorFlow2.0 CIFAR-10 Training')
    parser.add_argument('--model', default='resnet18', type=str, help='model type')
    parser.add_argument('--checkpoint_path', required=True, type=str, help="path to the model's checkpoint")
    args = parser.parse_args()
    args.model = args.model.lower()
    
    print('==> Building model...')
    model = utils.create_model(args.model, num_classes=10)
    utils.load_weights_to_model(model, args.checkpoint_path)
    utils.evaluate_model(model)
