import os

dataset_path = "datasets/PlantVillage"

classes = os.listdir(dataset_path)

print("Number of classes:", len(classes))

print("\nDisease Classes:")

for disease in classes:
    print(disease)