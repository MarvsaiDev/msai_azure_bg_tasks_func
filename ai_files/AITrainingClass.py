from io import BytesIO
import logging
import pandas as pd
import torch, os
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.metrics import confusion_matrix

from utils.RabbitMQ import publishMsgOnRabbitMQ
from utils.custom_ai.net_model import Attention, CustomClassifier, CustomClassifierEmb, CustomClassifierEmb2, CustomClassifierEmbLN, CustomClassifierNorm, EnsembleModel, Net2, Net2EmbNorm, Net4, ResidualBlock, weights_init
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(dataframe.columns[0], axis=1).values
        self.targets = dataframe.iloc[:, 0].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])


AIMODELS = {
    "ResidualBlock" : ResidualBlock,
    "CustomClassifier": CustomClassifier,
    "CustomClassifierEmbLN": CustomClassifierEmbLN,
    "CustomClassifierEmb2": CustomClassifierEmb2,
    "CustomClassifierEmb": CustomClassifierEmb,
    "CustomClassifierNorm": CustomClassifierNorm,
    "Attention": Attention,
    "Net4": Net4,
    "Net2EmbNorm": Net2EmbNorm,
    "Net2": Net2,
    "EnsembleModel": EnsembleModel
}

# when using multiple model, we are going to use this dict to save the model type name to training and optimization purposes
class TrainAIModel:
    if torch.cuda.is_available():
        device = "cuda"
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"

    def __init__(self, targetColumnName, df, encodedClasses = None, mode="inference", modelType="CustomClassifier"):
        self.df = df
        self.targetColumnName = targetColumnName
        self.cols = df.shape[1]
        self.input_size = self.cols if mode == "inference" else self.cols-1
        self.output_classes = len(encodedClasses) if mode == "inference" else len(df.iloc[:, 0].unique())
        self.classesWithEncoding = {} if encodedClasses is None else encodedClasses
        self.modelType = modelType
        self.cm = None
        self.accuray = 0
        self.loss = 0
        self.model = None
        self.optimizer = None
        self.isSameEncodedClasses = True


    def __encode_target__(self, encodedClasses, resume = False):
        if (resume):
            # Assuming 'targets' is your numpy array of targets
            encoder = LabelEncoder()

            originalColumn = self.df.iloc[:, 0].to_numpy()
            encodedColumn = encoder.fit_transform(self.df.iloc[:, 0])

            for i in range(len(originalColumn)):
                self.classesWithEncoding[encodedColumn[i]] = originalColumn[i]

            isSameEncodedClasses = True

            for k1, v1 in encodedClasses.items():
                foundClass = False
                for k2, v2 in self.classesWithEncoding.items():
                    if (v1 == v2):
                        foundClass = True
                        break
                
                if (foundClass == False):
                    isSameEncodedClasses = False
                    break
            
            if (len(encodedClasses.values()) == len(self.classesWithEncoding.values()) and isSameEncodedClasses):
                self.isSameEncodedClasses = True
            else:
                self.isSameEncodedClasses = False


            self.df.iloc[:, 0] = encodedColumn
        else:
            # Assuming 'targets' is your numpy array of targets
            encoder = LabelEncoder()

            originalColumn = self.df.iloc[:, 0].to_numpy()
            encodedColumn = encoder.fit_transform(self.df.iloc[:, 0])

            for i in range(len(originalColumn)):
                self.classesWithEncoding[encodedColumn[i]] = originalColumn[i]

            self.df.iloc[:, 0] = encodedColumn


    def __split_df__(self, df):
        # Split the data into training and validation sets
        self.train_df, self.val_df = train_test_split(df, test_size=0.1, random_state=42)

        # Create data loaders for training and validation sets
        self.train_dataset = MyDataset(self.train_df)
        self.val_dataset = MyDataset(self.val_df)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)


    def __define_model__(self):
        self.model = AIMODELS[self.modelType](input_size = self.input_size, output_size = self.output_classes)
        self.model.apply(weights_init)
        # Define a loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.true_labels = []
        self.pred_labels = []

    async def train_model(self, email, epochsNumbers):
        self.__encode_target__([], False)
        self.__split_df__(self.df)
        self.__define_model__()

        NO_EPOCHS= 10 if epochsNumbers is None or epochsNumbers < 1 else epochsNumbers
        # Training loop
        for epoch in range(NO_EPOCHS):  # number of epochs
            # this part is for training
            self.model.train()
            for features, targets in self.train_dataset:
                # Forward pass
                outputs = self.model(features.float())
                loss = self.criterion(outputs, targets.long())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = loss.item()
            logging.info(f'Epoch {epoch+1}, Loss: {train_loss}')


            # this part is for evaluation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features, targets in self.val_dataloader:
                    outputs = self.model(features.float())
                    loss = self.criterion(outputs, targets.long())
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    self.true_labels.append(targets)
                    self.pred_labels.append(predicted)

            # Compute the confusion matrix
            true_labelst = torch.cat(self.true_labels)
            pred_labelst = torch.cat(self.pred_labels)

            # When you need to use them as numpy arrays, you can move them to CPU
            true_labels_cpu = true_labelst.cpu().numpy()
            pred_labels_cpu = pred_labelst.cpu().numpy()
            cm = confusion_matrix(true_labels_cpu, pred_labels_cpu)

            self.cm = cm

            # Convert the confusion matrix to a DataFrame for easier visualization
            cm_df = pd.DataFrame(cm)
            # Compute the percentage confusion matrix
            cm_percentage = cm_df.div(cm_df.sum(axis=1), axis=0)
            # print(cm_percentage)
            print(cm_percentage.values.diagonal())

            accuracy = correct / total
            percentage = (epoch + 1) / NO_EPOCHS * 100

            await publishMsgOnRabbitMQ({
                "task": "training", 
                "condition": "continue", 
                "percentage": str(percentage), 
                "epoch": str(epoch + 1),
                "val_loss": str(loss.item()),
                "val_accuracy": str(accuracy),
                "train_loss": str(train_loss)
            }, email)

            self.accuray = accuracy
            self.loss = loss.item()

            logging.info(f'Epoch {epoch + 1}, Loss: {loss.item()}, Validation Accuracy: {accuracy}')

    async def __define_resume_model__(self, modelPath, container):
        # Initialize the model
        self.model = AIMODELS[self.modelType](input_size = self.input_size, output_size = self.output_classes)
        self.model.apply(weights_init)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.true_labels = []
        self.pred_labels = []

        # Load model and optimizer state
        model_data = await self.download_blob_func(container, os.path.join(modelPath, "model_data.pt"))
        optimizer_data = await self.download_blob_func(container, os.path.join(modelPath, "model_opti_data.pt"))
        state_dict = torch.load(model_data)

        if self.isSameEncodedClasses == False:
            logging.info("update only last layer.")
            num_features = self.model.fc.in_features

            # Create a new final layer and initialize
            self.model.fc = nn.Linear(num_features, len(self.df.iloc[:, 0].unique()))
            self.model.fc.apply(weights_init)

        else:
            self.model.load_state_dict(state_dict)

        
        self.optimizer.load_state_dict(torch.load(optimizer_data))

        logging.info("Model loaded and updated successfully!")

    async def download_blob_func(self, container, blob):
        stream = BytesIO()
        container.download_blob(blob).readinto(stream)
        stream.seek(0)
        return stream

    async def resume_train_model(self, email, epochsNumbers, modelPath, container, encodedClasses):
        self.__encode_target__(encodedClasses, resume=True)
        self.__split_df__(self.df)
        await self.__define_resume_model__(modelPath, container)

        NO_EPOCHS= 10 if epochsNumbers is None or epochsNumbers < 1 else epochsNumbers
        # Training loop
        for epoch in range(NO_EPOCHS):  # number of epochs
            # this part is for training
            self.model.train()
            for features, targets in self.train_dataset:
                # Forward pass
                outputs = self.model(features.float())
                loss = self.criterion(outputs, targets.long())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = loss.item()
            logging.info(f'Epoch {epoch+1}, Loss: {train_loss}')


            # this part is for evaluation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features, targets in self.val_dataloader:
                    outputs = self.model(features.float())
                    loss = self.criterion(outputs, targets.long())
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    self.true_labels.append(targets)
                    self.pred_labels.append(predicted)

            # Compute the confusion matrix
            true_labelst = torch.cat(self.true_labels)
            pred_labelst = torch.cat(self.pred_labels)

            # When you need to use them as numpy arrays, you can move them to CPU
            true_labels_cpu = true_labelst.cpu().numpy()
            pred_labels_cpu = pred_labelst.cpu().numpy()
            cm = confusion_matrix(true_labels_cpu, pred_labels_cpu)

            self.cm = cm

            # Convert the confusion matrix to a DataFrame for easier visualization
            cm_df = pd.DataFrame(cm)
            # Compute the percentage confusion matrix
            cm_percentage = cm_df.div(cm_df.sum(axis=1), axis=0)
            # print(cm_percentage)
            print(cm_percentage.values.diagonal())

            accuracy = correct / total
            percentage = (epoch + 1) / NO_EPOCHS * 100

            await publishMsgOnRabbitMQ({
                "task": "training", 
                "condition": "continue", 
                "percentage": str(percentage), 
                "epoch": str(epoch + 1),
                "val_loss": str(loss.item()),
                "val_accuracy": str(accuracy),
                "train_loss": str(train_loss)
            }, email)

            self.accuray = accuracy
            self.loss = loss.item()

            logging.info(f'Epoch {epoch + 1}, Loss: {loss.item()}, Validation Accuracy: {accuracy}')
