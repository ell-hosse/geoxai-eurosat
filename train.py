from dataset_loader import get_eurosat_dataloaders

train_loader, test_loader, class_names = get_eurosat_dataloaders()

print(f"Classes: {class_names}")
print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
