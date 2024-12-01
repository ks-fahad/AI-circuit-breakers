from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fine_tune import apply_lora, lora_circuit_breaker
from evaluate import evaluate_robustness_multiple_attacks
from attacks import get_adversarial_attacks
from utils import plot_robustness
from data_loader import load_json_dataset
from huggingface_hub import login

def main():
    train_file_path = 'data/circuit_breakers_train.json'
    val_file_path = 'data/circuit_breakers_val.json'
    
    train_dataset, val_dataset = load_json_dataset(train_file_path, val_file_path)
    login(token="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")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    model = apply_lora(model)

    print("Evaluating initial model robustness...")
    initial_robustness = evaluate_robustness_multiple_attacks(
        model, tokenizer, val_dataset['prompt'], attacks=get_adversarial_attacks()
    )
    print(f"Initial Robustness: {initial_robustness}")

    plot_robustness(initial_robustness, title="Initial Model Robustness with Multiple Attacks")

    print("Fine-tuning the model with LoRA and circuit breaker...")
    fine_tuned_model = lora_circuit_breaker(
        model, train_dataset=train_dataset, eval_dataset=val_dataset, output_dir = "./out/Mistral-7b_CB",
        evaluate_robustness_fn=lambda model: evaluate_robustness_multiple_attacks(model, tokenizer, val_dataset['prompt'], attacks=get_adversarial_attacks())
    )

    print("Evaluating fine-tuned model robustness...")
    final_robustness = evaluate_robustness_multiple_attacks(
        fine_tuned_model, tokenizer, val_dataset['prompt'], attacks=get_adversarial_attacks()
    )
    print(f"Final Robustness after Fine-Tuning: {final_robustness}")

    plot_robustness(final_robustness, title="Fine-Tuned Model Robustness with Multiple Attacks")

if __name__ == "__main__":
    main()
