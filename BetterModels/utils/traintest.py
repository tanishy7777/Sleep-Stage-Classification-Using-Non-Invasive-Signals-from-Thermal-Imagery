

def train_loop(model, num_epochs, criterion, optimizer, train_loader, device):
    train_results = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for sequences, tsfel_features, labels in train_loader:
            sequences = sequences.to(device)
            tsfel_features = tsfel_features.to(device)
            labels = labels.to(device).reshape(-1, 1)

            optimizer.zero_grad()
            outputs = model(sequences, tsfel_features)  # Pass both inputs to the model
            loss = criterion(outputs, labels)
            train_results.append((labels, outputs))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}", end='\r')

        avg_loss = total_loss / len(train_loader)
        print(f"\nTraining Loss for Epoch {epoch+1}: {avg_loss:.4f}")
    return train_results

def test_loop(model, criterion, test_loader, device):
    test_results = []
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for sequences, tsfel_features, labels in test_loader:
            sequences = sequences.to(device)
            tsfel_features = tsfel_features.to(device)
            labels = labels.to(device).reshape(-1, 1)

            outputs = model(sequences, tsfel_features)  # Pass both inputs to the model
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs)
            predicted_labels = (preds > 0.5).float()
            test_results.append((labels, preds))

            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}")
    return test_results


