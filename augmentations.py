from torchvision import transforms

#to po to zeby miec wiecej danych do treningu, 
# po prostu modyfikujemy troche zdjecie i mamy nowe dane treningowe
def get_train_transforms():

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        
        #obrot w lewo/prawo losowy
        transforms.RandomRotation(degrees=10), 
        
        #zoom i przesuniecie
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        
        #zmiana kontrastu
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

#to do ewaluacji i testow, tu nic nie zmieniamy
def get_eval_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])