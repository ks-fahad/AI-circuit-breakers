from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import PyTorchClassifier
import torch

def evaluate_robustness_multiple_attacks(model, tokenizer, test_texts, attacks=[FastGradientMethod, ProjectedGradientDescent, CarliniL2Method]):
    adversarial_accuracies = {}

    # Convert the model to an ART classifier
    classifier = PyTorchClassifier(model=model,
                                   loss=torch.nn.CrossEntropyLoss(),
                                   optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                                   input_shape=(1, 224, 224), 
                                   nb_classes=2) 

    for attack in attacks:
        print(f"Evaluating with {attack.__name__}...")
        attack_method = attack(classifier)
        adversarial_data = attack_method.generate(x=test_texts)

        # Evaluate the model performance on adversarial examples
        accuracy = classifier.evaluate(test_texts, adversarial_data)
        adversarial_accuracies[attack.__name__] = accuracy
    
    return adversarial_accuracies
