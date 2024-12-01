from peft import get_peft_model, LoraConfig
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from tqdm import tqdm
from evaluate import evaluate_robustness_multiple_attacks

def apply_lora(model: torch.nn.Module, lora_r: int = 8, lora_alpha: int = 16, target_modules: list = ['q_proj', 'v_proj']):
    lora_config = LoraConfig(
        r=lora_r,
        alpha=lora_alpha,
        target_modules=target_modules,
        dropout=0.1,  
        bias='none' 
    )
    model = get_peft_model(model, lora_config)
    return model

def lora_circuit_breaker(model, train_dataset, eval_dataset, output_dir, evaluate_robustness_fn, max_epochs=3, stop_threshold=0.5):
    model = apply_lora(model)
    
    training_args = TrainingArguments(
        output_dir=output_dir,          
        evaluation_strategy="epoch",    
        learning_rate= 5e-4,   
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,  
        num_train_epochs=5,            
        weight_decay=0.01,            
        logging_dir='./logs',          
        logging_steps=10,            
        save_steps=10,              
        load_best_model_at_end=True,    
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=evaluate_robustness_multiple_attacks
    )

    for epoch in tqdm(range(max_epochs)):
        trainer.train() 

        print(f"Evaluating adversarial robustness after epoch {epoch + 1}...")
        robustness = evaluate_robustness_fn(model)
        print(f"Adversarial robustness (accuracy): {robustness}")
        
        if robustness < stop_threshold:
            print(f"Circuit breaker triggered: robustness too low ({robustness}). Stopping training.")
            break

    return model
