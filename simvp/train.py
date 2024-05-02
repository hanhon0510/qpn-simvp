# train.py
import torch
from torch.utils.data import DataLoader
from dataset import TIFDataset
# from models import SimVP
from simvp import SimVP
import torch.optim as optim
import os

def main():
    # root_dir = '/data/DATA_SV/Precipitation/Radar/2019/10/01' 
    # root_dir = '../data/DATA_SV/Precipitation/Radar/2019/10/01'
    # root_dir = r'D:\Workspace\Projects\qpn-simvp\data\DATA_SV\Precipitation\Radar\2019\10\01'

    root_dir = r'D:\Workspace\Projects\qpn-simvp\data\DATA_SV\Precipitation\Radar\2019\10'
    save_dir = r'D:\Workspace\Projects\qpn-simvp\simvp\pth'  # Define the directory to save your .pth files
    os.makedirs(save_dir, exist_ok=True)
    print(f"Using root_dir: {root_dir}")
    print(f"Does root_dir exist? {'Yes' if os.path.exists(root_dir) else 'No'}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dataset = TIFDataset(root_dir, sequence_length=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size of 1 given the sequence nature

    shape_in = (12, 1, 304, 304)  # Adjust accordingly
    model = SimVP(shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8)
    model = model.to(device)
    #train(model, dataloader, device)
    test(model)

def train(model, dataloader, device, save_dir):

    # Model initialization, loss function, and optimizer definition here
    criterion = torch.nn.MSELoss()  # Example: MSE loss for prediction tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)  

    # Training loop
    num_epochs = 10  # Define the number of epochs you want to train for
    time_step_buffer = []  # Initialize the buffer outside the epoch loop
    sequence_length = 12  # Define the desired sequence length

    def check_nan(tensor, name="Tensor"):
      """Utility function to check for NaN values in a tensor."""
      isnan = torch.isnan(tensor).any().item()
      print(f"{name} contains NaN: {isnan}")
      return isnan
    
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Ensure inputs are on the correct device
            inputs, targets = inputs.to(device), targets.to(device)

            if check_nan(inputs, "Inputs"):
              print("NaN detected in inputs, skipping this batch.")
              continue

            # Accumulate time steps in the buffer
            time_step_buffer.append(inputs)

            # Proceed only if we have accumulated enough time steps
            if len(time_step_buffer) == sequence_length:
                # Stack the accumulated time steps along a new dimension
                sequence_input = torch.stack(time_step_buffer, dim=1)  # Resulting shape should be [1, sequence_length, 1, 90, 250]
                sequence_input = sequence_input.squeeze(2)  # Adjust the squeezing depending on your actual data shape
                
                # Ensure the sequence is on the correct device
                sequence_input = sequence_input.to(device)

                # Simulate targets for now; adjust as necessary for your application
                # Here, just for the sake of having a target of matching dimensions
                targets = torch.randn_like(sequence_input)  # Creating dummy targets of the same shape as sequence_input
                targets = targets.to(device)

                optimizer.zero_grad()

                # print(f"Input shape before model: {sequence_input.shape}")  # Ensure this matches your model's expected input
                sequence_input = sequence_input.unsqueeze(2)  # Add a singleton channel dimension
                # print(f"Adjusted input shape before model: {sequence_input.shape}")
                outputs = model(sequence_input)

                if check_nan(outputs, "Outputs"):
                  print("NaN detected in model outputs, skipping this batch.")
                  continue

                loss = criterion(outputs, targets)  # Compute loss
                if check_nan(loss, "Loss"):
                  print("NaN detected in loss, skipping this batch.")
                  continue
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters

                # Clear the buffer for the next sequence
                time_step_buffer.clear()

                if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}')

        # Save the model after each epoch
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved for epoch {epoch} at {model_save_path}")
        # torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
        # print(f'Model saved for epoch {epoch}')

def test(model):
  torch.load_state_dict(torch.load(model_save_path))

  if __name__ == '__main__':
    main()

def predict(model, input_sequence):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Assuming input_sequence is a tensor of shape [3, channels, height, width]
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
        input_sequence = input_sequence.to(device)
        output = model(input_sequence)
    return output