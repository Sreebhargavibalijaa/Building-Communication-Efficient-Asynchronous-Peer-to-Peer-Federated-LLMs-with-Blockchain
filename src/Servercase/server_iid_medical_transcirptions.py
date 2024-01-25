
from collections import OrderedDict
import os
import random
import warnings

import flwr as fl
import torch

from torch.utils.data import DataLoader

from datasets import load_dataset
from evaluate import load as load_metric

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import logging

"""Next we will set some global variables and disable some of the logging to clear out our output."""

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore')

DEVICE = torch.device("cpu")
CHECKPOINT = "dmis-lab/biobert-v1.1"  # transformer model checkpoint
NUM_CLIENTS = 5
NUM_ROUNDS = 20
import psutil
import time
"""## Standard Hugging Face workflow

### Handling the data

To fetch the IMDB dataset, we will use Hugging Face's `datasets` library. We then need to tokenize the data and create `PyTorch` dataloaders, this is all done in the `load_data` function:
"""
before_communication_cpu_percent = psutil.cpu_percent()
current_process = psutil.Process()

memory_info_after = current_process.memory_info()
start = time.time()

def load_data():
    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("bhargavi909/Medical_Transcriptions_upsampled")
    raw_datasets = raw_datasets.shuffle(seed=42)

    # remove unnecessary data split

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["description"], truncation=True)

    # Select 20 random samples to reduce the computation cost
    train_population = random.sample(range(len(raw_datasets["train"])), 500)
    test_population = random.sample(range(len(raw_datasets["test"])), 500)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(train_population)
    tokenized_datasets["test"] = tokenized_datasets["test"].select(test_population)
    tokenized_datasets = tokenized_datasets.remove_columns("Unnamed: 0")

    tokenized_datasets = tokenized_datasets.remove_columns("description")
    tokenized_datasets = tokenized_datasets.rename_column("medical_specialty", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader

"""### Training and testing the model

Once we have a way of creating our trainloader and testloader, we can take care of the training and testing. This is very similar to any `PyTorch` training or testing loop:
"""

def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

"""### Creating the model itself

To create the model itself, we will just load the pre-trained alBERT model using Hugging Face’s `AutoModelForSequenceClassification` :
"""

net = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT, num_labels=40
).to(DEVICE)

"""## Federating the example

The idea behind Federated Learning is to train a model between multiple clients and a server without having to share any data. This is done by letting each client train the model locally on its data and send its parameters back to the server, which then aggregates all the clients’ parameters together using a predefined strategy. This process is made very simple by using the [Flower](https://github.com/adap/flower) framework. If you want a more complete overview, be sure to check out this guide: [What is Federated Learning?](https://flower.dev/docs/tutorial/Flower-0-What-is-FL.html)

### Creating the IMDBClient

To federate our example to multiple clients, we first need to write our Flower client class (inheriting from `flwr.client.NumPyClient`). This is very easy, as our model is a standard `PyTorch` model:
"""

class IMDBClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.net, self.trainloader, epochs=1)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy), "loss": float(loss)}

"""The `get_parameters` function lets the server get the client's parameters. Inversely, the `set_parameters` function allows the server to send its parameters to the client. Finally, the `fit` function trains the model locally for the client, and the `evaluate` function tests the model locally and returns the relevant metrics.

### Generating the clients

In order to simulate the federated setting we need to provide a way to instantiate clients for our simulation. Here, it is very simple as every client will hold the same piece of data (this is not realistic, it is just used here for simplicity sakes).
"""

trainloader, testloader = load_data()
def client_fn(cid):
  return IMDBClient(net, trainloader, testloader)

"""## Starting the simulation

We now have all the elements to start our simulation. The `weighted_average` function is there to provide a way to aggregate the metrics distributed amongst the clients (basically to display a nice average accuracy at the end of the training). We then define our strategy (here `FedAvg`, which will aggregate the clients weights by doing an average).

Finally, `start_simulation` is used to start the training.
"""

def weighted_average(metrics):
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  losses = [num_examples * m["loss"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]
  return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0},
    ray_init_args={"log_to_driver": False, "num_cpus": 1, "num_gpus": 0}
)

"""Note that this is a very basic example, and a lot can be added or modified, it was just to showcase how simply we could federate a Hugging Face workflow using Flower. The number of clients and the data samples are intentionally very small in order to quickly run inside Colab, but keep in mind that everything can be tweaked and extended."""
after_communication_cpu_percent = psutil.cpu_percent()
current_process = psutil.Process()

memory_info_before = current_process.memory_info()

# Calculate the communication overhead
cpu_overhead = after_communication_cpu_percent - before_communication_cpu_percent
memory_overhead =(memory_info_after.rss - memory_info_before.rss) / (1024 ** 3)  # Convert bytes to GB
end = time.time()

print(f"CPU Overhead: {cpu_overhead}%")
print(f"Memory Usage: {memory_overhead:.2f} GB")
print(f"Latency: {(end-start)/60} min")


