import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from helper_logger import DataLogger
from helper_tester import ModelTesterMetrics
from dataset import SimpleTorchDataset  # Assurez-vous que dataset.py est dans le même dossier
from torchvision import transforms
from model_vgg import VGGCustom  # Assurez-vous que model_vgg est dans le même dossier

# Fixer les seeds pour la reproductibilité
SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_epochs = 64
batch_size = 16

if __name__ == "__main__":

    print("| Pytorch Model Training!")
    
    print("| Total Epochs:", total_epochs)
    print("| Batch Size:", batch_size)
    print("| Device:", device)

    # Initialiser le logger pour enregistrer les résultats
    logger = DataLogger("SimpleFlowersClassificationVGG")
    metrics = ModelTesterMetrics()

    # Définir la fonction de perte et l'activation
    metrics.loss = torch.nn.CrossEntropyLoss()  # Pour classification multi-classes
    metrics.activation = torch.nn.Softmax(dim=1)

    # Charger le modèle VGGCustom pour la classification de fleurs (3 classes dans ce cas)
    model = VGGCustom(output_classes=3).to(device)  # 3 classes pour le dataset Flowers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Transformations pour les données d'entraînement et de validation
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),  # Convertir l'image en tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Convertir l'image en tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Chargement des datasets Flowers
    validation_dataset = SimpleTorchDataset('./AsFlowers/val', aug=val_transforms)
    training_dataset = SimpleTorchDataset('./AsFlowers/train', aug=train_transforms)
    testing_dataset = SimpleTorchDataset('./AsFlowers/test', aug=val_transforms)

    # DataLoaders pour les datasets
    validation_datasetloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    training_datasetloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_datasetloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)

    # Boucle d'entraînement et d'évaluation
    for current_epoch in range(total_epochs):
        print("Epoch:", current_epoch)
        
        # Phase d'entraînement
        model.train()  # Mode entraînement
        metrics.reset()  # Réinitialiser les métriques

        for (image, label) in tqdm(training_datasetloader, desc="Training:"):

            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()  # Remise à zéro des gradients
            output = model(image)
            loss = metrics.compute(output, label)
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des poids
            
        training_mean_loss = metrics.average_loss()
        training_mean_accuracy = metrics.average_accuracy()

        # Phase d'évaluation
        model.eval()  # Mode évaluation
        metrics.reset()  # Réinitialiser les métriques

        with torch.no_grad():
            for (image, label) in tqdm(validation_datasetloader, desc="Testing:"):
                
                image = image.to(device)
                label = label.to(device)

                output = model(image)
                metrics.compute(output, label)

        evaluation_mean_loss = metrics.average_loss()
        evaluation_mean_accuracy = metrics.average_accuracy()

        # Enregistrer les résultats dans le logger
        logger.append(
            current_epoch,
            training_mean_loss,
            training_mean_accuracy,
            evaluation_mean_loss,
            evaluation_mean_accuracy
        )

        # Sauvegarder le meilleur modèle
        if logger.current_epoch_is_best:
            print("> Latest Best Epoch:", logger.best_accuracy())
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            state_dictonary = {
                "model_state": model_state,
                "optimizer_state": optimizer_state
            }
            torch.save(
                state_dictonary, 
                logger.get_filepath("best_checkpoint.pth")
            )

        logger.save()
        print("")

    print("| Training Complete, Loading Best Checkpoint")
    
    # Charger l'état du modèle sauvegardé
    state_dictonary = torch.load(
        logger.get_filepath("best_checkpoint.pth"), 
        map_location=device
    )
    model.load_state_dict(state_dictonary['model_state'])
    model = model.to(device)
    
    # Tester le modèle final sur le dataset de test
    model.eval()  # Mode évaluation
    metrics.reset()  # Réinitialiser les métriques

    for (image, label) in tqdm(testing_datasetloader):
        
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        metrics.compute(output, label)

    testing_mean_loss = metrics.average_loss()
    testing_mean_accuracy = metrics.average_accuracy()

    print("")
    logger.write_text(f"# Final Testing Loss: {testing_mean_loss}")
    logger.write_text(f"# Final Testing Accuracy: {testing_mean_accuracy}")
    logger.write_text(f"# Report:")
    logger.write_text(metrics.report())
    logger.write_text(f"# Confusion Matrix:")
    logger.write_text(metrics.confusion())
    print("")
