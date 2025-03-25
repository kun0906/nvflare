
import torch
import torch.nn.functional as F

# Create a tensor for the softmax probabilities as mentioned
P = torch.tensor([1.0, 1.97e-48, 1.04e-91])

# Compute log_softmax
log_softmax_probs = F.log_softmax(P, dim=0)
print(log_softmax_probs)


#
#
# import sys
#
# # Assuming two parameters are passed
# DISTILL_WEIGHT = sys.argv[1]  # First parameter
# LOCAL_EPOCHS = sys.argv[2]  # Second parameter
#
# print(f"DISTILL_WEIGHT 1: {DISTILL_WEIGHT}")
# print(f"LOCAL_EPOCHS 2: {LOCAL_EPOCHS}")
print('demo.py')

import argparse

# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="Test Demo Script")

    # Add arguments to be parsed
    parser.add_argument('-v', '--distill_weight', type=float, required=True,
                        help="The distillation weight (float).")
    parser.add_argument('-n', '--epochs', type=int, required=True,
                        help="The number of epochs (integer).")

    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()

    # Access the arguments
    distill_weight = args.distill_weight
    epochs = args.epochs

    # For testing, print the parsed parameters
    print(f"Distill Weight: {distill_weight}")
    print(f"Epochs: {epochs}")

    # You can now use distill_weight and epochs in your script logic
    # Example:
    # model = your_model_setup_function()
    # train_model(model, distill_weight=distill_weight, epochs=epochs)

    # For now, we'll just simulate training
    print(f"Simulating training with distill_weight = {distill_weight} for {epochs} epochs.")
    # Add your model training logic here

    # for i in range(1000000000):
    #     a = i % 2


