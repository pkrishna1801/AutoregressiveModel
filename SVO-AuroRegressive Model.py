import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from sklearn.model_selection import train_test_split

# ============= Dataset Class for CSV Data =============

class GrammarToNaturalDataset(Dataset):
    """Dataset for converting grammatical annotations to natural language from CSV"""
    
    def __init__(self, csv_path, tokenizer, max_length=128, model_type='causal', 
                 input_column='input', output_column='output'):
        """
        Args:
            csv_path: Path to CSV file with columns for input and output
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            model_type: 'causal' for GPT-style or 'seq2seq' for T5-style
            input_column: Name of column containing grammar annotations
            output_column: Name of column containing natural sentences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
        # Load data from CSV
        self.df = pd.read_csv(csv_path)
        
        # Handle missing values
        self.df = self.df.dropna(subset=[input_column, output_column])
        
        # Store columns
        self.inputs = self.df[input_column].tolist()
        self.outputs = self.df[output_column].tolist()
        
        print(f"Loaded {len(self.inputs)} examples from {csv_path}")
        print(f"Sample input: {self.inputs[0]}")
        print(f"Sample output: {self.outputs[0]}")
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = str(self.inputs[idx])
        output_text = str(self.outputs[idx])
        
        if self.model_type == 'seq2seq':
            # For T5 or BART models
            input_formatted = f"convert grammar to text: {input_text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_formatted,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize output (labels)
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(
                    output_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
            # Replace padding token id's of the labels by -100
            targets['input_ids'][targets['input_ids'] == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets['input_ids'].squeeze()
            }
        
        else:  # causal LM (GPT-style)
            # Format for instruction-following
            prompt = f"### Instruction: Convert the grammatical annotation to a natural sentence.\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            
            # Add end token
            if self.tokenizer.eos_token:
                prompt += self.tokenizer.eos_token
            
            # Tokenize the full prompt
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # For causal LM, labels are the same as input_ids
            labels = encoded['input_ids'].clone()
            
            # Mask the instruction part (only train on response)
            response_start = prompt.find("### Response:\n") + len("### Response:\n")
            response_start_token = len(self.tokenizer.encode(prompt[:response_start]))
            labels[0, :response_start_token] = -100
            
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': labels.squeeze()
            }

# ============= Data Preparation Utilities =============

class DataPreprocessor:
    """Preprocess and validate CSV data"""
    @staticmethod
    def split_csv_data(csv_path, output_dir='./', test_size=0.1, val_size=0.1, 
                    input_col='input', output_col='output', random_state=42):
        """Split CSV data into train, validation, and test sets"""
        
        import os
        
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[input_col, output_col])
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Save splits
        train_df.to_csv(f"{output_dir}/train.csv", index=False)
        val_df.to_csv(f"{output_dir}/val.csv", index=False)
        test_df.to_csv(f"{output_dir}/test.csv", index=False)
        
        print(f"Data split completed:")
        print(f"- Train: {len(train_df)} samples -> {output_dir}/train.csv")
        print(f"- Validation: {len(val_df)} samples -> {output_dir}/val.csv")
        print(f"- Test: {len(test_df)} samples -> {output_dir}/test.csv")
        
        return train_df, val_df, test_df

    @staticmethod
    def validate_csv(csv_path, input_col='input', output_col='output'):
        """Validate CSV file structure and content"""
        try:
            df = pd.read_csv(csv_path)
            
            # Check required columns exist
            if input_col not in df.columns:
                raise ValueError(f"Column '{input_col}' not found in CSV")
            if output_col not in df.columns:
                raise ValueError(f"Column '{output_col}' not found in CSV")
            
            # Check for empty values
            null_inputs = df[input_col].isnull().sum()
            null_outputs = df[output_col].isnull().sum()
            
            print(f"CSV Validation Report:")
            print(f"- Total rows: {len(df)}")
            print(f"- Null inputs: {null_inputs}")
            print(f"- Null outputs: {null_outputs}")
            print(f"- Valid rows: {len(df) - max(null_inputs, null_outputs)}")
            
            return True
            
        except Exception as e:
            print(f"Error validating CSV: {e}")
            return False
    
    # @staticmethod
    # def split_csv_data(csv_path, output_dir='./', test_size=0.1, val_size=0.1, 
    #                    input_col='input', output_col='output', random_state=42):
    #     """Split CSV data into train, validation, and test sets"""
        
    #     df = pd.read_csv(csv_path)
    #     df = df.dropna(subset=[input_col, output_col])
        
    #     # First split: separate test set
    #     train_val_df, test_df = train_test_split(
    #         df, test_size=test_size, random_state=random_state
    #     )
        
    #     # Second split: separate train and validation
    #     val_size_adjusted = val_size / (1 - test_size)
    #     train_df, val_df = train_test_split(
    #         train_val_df, test_size=val_size_adjusted, random_state=random_state
    #     )
        
    #     # Save splits
    #     train_df.to_csv(f"{output_dir}/train.csv", index=False)
    #     val_df.to_csv(f"{output_dir}/val.csv", index=False)
    #     test_df.to_csv(f"{output_dir}/test.csv", index=False)
        
    #     print(f"Data split completed:")
    #     print(f"- Train: {len(train_df)} samples -> {output_dir}/train.csv")
    #     print(f"- Validation: {len(val_df)} samples -> {output_dir}/val.csv")
    #     print(f"- Test: {len(test_df)} samples -> {output_dir}/test.csv")
        
    #     return train_df, val_df, test_df
    
    @staticmethod
    def analyze_grammar_patterns(csv_path, input_col='input'):
        """Analyze grammar patterns in the dataset"""
        df = pd.read_csv(csv_path)
        
        # Extract all grammar tags
        all_tags = []
        for input_str in df[input_col].dropna():
            tags = re.findall(r'\[([^\]]+)\]', str(input_str))
            all_tags.extend(tags)
        
        # Count tag frequencies
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        print("\nGrammar Tag Analysis:")
        print("-" * 40)
        for tag, count in tag_counts.most_common():
            print(f"{tag:15} : {count:5} occurrences")
        
        return tag_counts

# ============= Model Training Functions =============

def train_t5_model(
    train_csv_path, 
    val_csv_path=None, 
    model_name='t5-small',
    output_dir='./grammar-t5-model',
    num_epochs=10,
    batch_size=8,
    learning_rate=3e-4,
    input_column='input',
    output_column='output'
):
    """Train T5 model for grammar-to-natural conversion using CSV data"""
    
    print(f"\n{'='*50}")
    print(f"Training T5 Model: {model_name}")
    print(f"{'='*50}")
    
    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Prepare datasets
    train_dataset = GrammarToNaturalDataset(
        train_csv_path, 
        tokenizer, 
        model_type='seq2seq',
        input_column=input_column,
        output_column=output_column
    )
    
    val_dataset = None
    if val_csv_path:
        val_dataset = GrammarToNaturalDataset(
            val_csv_path, 
            tokenizer, 
            model_type='seq2seq',
            input_column=input_column,
            output_column=output_column
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy='steps' if val_dataset else 'no',
        eval_steps=500 if val_dataset else None,
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model='loss',
        greater_is_better=False,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_model_path = f'{output_dir}/final_model'
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    return model, tokenizer

def train_gpt_model(
    train_csv_path,
    val_csv_path=None,
    model_name='gpt2',
    output_dir='./grammar-gpt-model',
    num_epochs=5,
    batch_size=4,
    learning_rate=5e-5,
    input_column='input',
    output_column='output'
):
    """Train GPT-style model for grammar-to-natural conversion using CSV data"""
    
    print(f"\n{'='*50}")
    print(f"Training GPT Model: {model_name}")
    print(f"{'='*50}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for our grammar tags
    special_tokens = {
        'additional_special_tokens': [
            '[subj]', '[verb]', '[obj]', '[location]', '[time]', '[tense]',
            '[adjective]', '[adverb]', '[pronoun]', '[preposition]',
            '### Instruction:', '### Input:', '### Response:'
        ]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added {num_added_tokens} special tokens")
    
    # Prepare datasets
    train_dataset = GrammarToNaturalDataset(
        train_csv_path,
        tokenizer,
        model_type='causal',
        input_column=input_column,
        output_column=output_column
    )
    
    val_dataset = None
    if val_csv_path:
        val_dataset = GrammarToNaturalDataset(
            val_csv_path,
            tokenizer,
            model_type='causal',
            input_column=input_column,
            output_column=output_column
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy='steps' if val_dataset else 'no',
        eval_steps=500 if val_dataset else None,
        save_steps=1000,
        save_total_limit=3,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True if val_dataset else False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_model_path = f'{output_dir}/final_model'
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    return model, tokenizer

def fine_tune_llama_lora(
    train_csv_path,
    val_csv_path=None,
    model_name='meta-llama/Llama-2-7b-hf',
    output_dir='./grammar-llama-lora',
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4,
    input_column='input',
    output_column='output'
):
    """Fine-tune Llama model using LoRA for efficient training"""
    
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    import torch
    
    print(f"\n{'='*50}")
    print(f"Fine-tuning Llama with LoRA: {model_name}")
    print(f"{'='*50}")
    
    # Quantization config for 4-bit loading (saves memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset = GrammarToNaturalDataset(
        train_csv_path,
        tokenizer,
        model_type='causal',
        max_length=256,
        input_column=input_column,
        output_column=output_column
    )
    
    val_dataset = None
    if val_csv_path:
        val_dataset = GrammarToNaturalDataset(
            val_csv_path,
            tokenizer,
            model_type='causal',
            max_length=256,
            input_column=input_column,
            output_column=output_column
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=50,
        eval_strategy='steps' if val_dataset else 'no',
        eval_steps=200 if val_dataset else None,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True if val_dataset else False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting LoRA fine-tuning...")
    trainer.train()
    
    # Save LoRA weights
    trainer.save_model(output_dir)
    print(f"\nLoRA weights saved to: {output_dir}")
    
    return model, tokenizer

# ============= Inference Functions =============

class GrammarToNaturalInference:
    """Inference class for converting grammar annotations to natural language"""
    
    def __init__(self, model_path, model_type='t5', device=None):
        """
        Initialize inference model
        
        Args:
            model_path: Path to saved model
            model_type: 't5', 'gpt', or 'llama-lora'
            device: Device to run inference on (cuda/cpu)
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        elif model_type == 'llama-lora':
            from peft import PeftModel
            # Load base model and LoRA weights
            base_model_name = 'meta-llama/Llama-2-7b-hf'  # Adjust as needed
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:  # GPT-style
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, grammar_input, max_length=50, temperature=0.7, top_p=0.9, num_beams=4):
        """Generate natural language from grammar annotation"""
        
        if self.model_type == 't5':
            input_text = f"convert grammar to text: {grammar_input}"
            inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        else:  # GPT-style or Llama
            if self.model_type == 'llama-lora':
                prompt = f"### Instruction: Convert the grammatical annotation to a natural sentence.\n\n### Input:\n{grammar_input}\n\n### Response:\n"
            else:
                prompt = f"### Instruction: Convert the grammatical annotation to a natural sentence.\n\n### Input:\n{grammar_input}\n\n### Response:\n"
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in generated:
                return generated.split("### Response:")[-1].strip()
            return generated
    
    def batch_generate(self, grammar_inputs, batch_size=8, **kwargs):
        """Generate natural language for multiple inputs"""
        results = []
        
        for i in range(0, len(grammar_inputs), batch_size):
            batch = grammar_inputs[i:i+batch_size]
            for input_text in batch:
                output = self.generate(input_text, **kwargs)
                results.append(output)
        
        return results

# ============= Evaluation Functions =============

def evaluate_model(model_path, test_csv_path, model_type='t5', 
                  input_column='input', output_column='output'):
    """Evaluate model performance on test set"""
    
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    import numpy as np
    
    print(f"\n{'='*50}")
    print(f"Evaluating Model: {model_path}")
    print(f"{'='*50}")
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    test_df = test_df.dropna(subset=[input_column, output_column])
    
    # Initialize inference
    inference = GrammarToNaturalInference(model_path, model_type)
    
    # Generate predictions
    predictions = []
    references = []
    
    print("Generating predictions...")
    for idx, row in test_df.iterrows():
        input_text = str(row[input_column])
        reference = str(row[output_column])
        
        prediction = inference.generate(input_text)
        predictions.append(prediction)
        references.append(reference)
        
        if idx < 5:  # Show first 5 examples
            print(f"\nExample {idx + 1}:")
            print(f"Input:      {input_text}")
            print(f"Predicted:  {prediction}")
            print(f"Reference:  {reference}")
    
    # Calculate metrics
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    
    # Calculate BLEU scores
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        bleu = sentence_bleu([ref.split()], pred.split())
        bleu_scores.append(bleu)
    
    avg_bleu = np.mean(bleu_scores)
    
    print(f"\n{'='*40}")
    print(f"Evaluation Results:")
    print(f"{'='*40}")
    print(f"BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
    
    return {
        'bleu': avg_bleu,
        'rouge': rouge_scores,
        'predictions': predictions,
        'references': references
    }

# ============= Main Training Pipeline =============

def main():
    """Main training pipeline using CSV data"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Grammar-to-Natural Language Model')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to CSV file with training data')
    parser.add_argument('--model_type', type=str, default='t5', choices=['t5', 'gpt', 'llama-lora'], 
                       help='Model type to train')
    parser.add_argument('--model_name', type=str, default=None, help='Base model name')
    parser.add_argument('--output_dir', type=str, default='./model_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--input_column', type=str, default='input', help='CSV column name for input')
    parser.add_argument('--output_column', type=str, default='output', help='CSV column name for output')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    
    args = parser.parse_args()
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    # Validate CSV
    print("Validating CSV file...")
    if not DataPreprocessor.validate_csv(args.data_csv, args.input_column, args.output_column):
        print("CSV validation failed. Please check your file.")
        return
    
    # Analyze grammar patterns
    print("\nAnalyzing grammar patterns...")
    DataPreprocessor.analyze_grammar_patterns(args.data_csv, args.input_column)
    
    # Split data
    print("\nSplitting data into train/val/test sets...")
    train_df, val_df, test_df = DataPreprocessor.split_csv_data(
        args.data_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        input_col=args.input_column,
        output_col=args.output_column
    )
    
    # Set model name defaults
    if args.model_name is None:
        model_names = {
            't5': 't5-small',
            'gpt': 'gpt2',
            'llama-lora': 'meta-llama/Llama-2-7b-hf'
        }
        args.model_name = model_names[args.model_type]
    
    # Train model
    train_csv = f"{args.output_dir}/train.csv"
    val_csv = f"{args.output_dir}/val.csv"
    test_csv = f"{args.output_dir}/test.csv"
    
    if args.model_type == 't5':
        model, tokenizer = train_t5_model(
            train_csv,
            val_csv,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            input_column=args.input_column,
            output_column=args.output_column
        )
        model_path = f"{args.output_dir}/final_model"
        
    elif args.model_type == 'gpt':
        model, tokenizer = train_gpt_model(
            train_csv,
            val_csv,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            input_column=args.input_column,
            output_column=args.output_column
        )
        model_path = f"{args.output_dir}/final_model"
        
    else:  # llama-lora
        model, tokenizer = fine_tune_llama_lora(
            train_csv,
            val_csv,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            input_column=args.input_column,
            output_column=args.output_column
        )
        model_path = args.output_dir
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    evaluation_results = evaluate_model(
        model_path,
        test_csv,
        model_type=args.model_type,
        input_column=args.input_column,
        output_column=args.output_column
    )
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("="*50)

if __name__ == "__main__":
    # If running without command line arguments, use this example
    import sys
    
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python train.py --data_csv data.csv --model_type t5 --epochs 10")
        print("\nOr run the quick test below:")
        
        # Create sample CSV for testing
        sample_data = {
            'input': [
                '[subj]Cat[verb]eat[obj]mouse[location]house[tense]past',
                '[subj]Dog[verb]run[location]park[tense]present',
                '[subj]They[verb]build[obj]bridge[tense]future',
                '[subj]Bird[verb]sing[location]tree[tense]present',
                '[subj]I[verb]write[obj]letter[location]office[tense]past',
                '[subj]She[verb]read[obj]book[tense]present',
                '[subj]We[verb]play[obj]game[location]home[tense]future',
                '[subj]Teacher[verb]teach[obj]students[location]school[tense]present',
                '[subj]Children[verb]swim[location]pool[tense]past',
                '[subj]He[verb]cook[obj]dinner[location]kitchen[tense]present'
            ],
            'output': [
                'The cat ate the mouse in the house.',
                'The dog runs in the park.',
                'They will build a bridge.',
                'The bird sings in the tree.',
                'I wrote a letter in the office.',
                'She reads the book.',
                'We will play a game at home.',
                'The teacher teaches students at school.',
                'The children swam in the pool.',
                'He cooks dinner in the kitchen.'
            ]
        }
        
        # Save sample data
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv('sample_data.csv', index=False)
        print("\nCreated sample_data.csv for testing")
        print("Run: python train.py --data_csv sample_data.csv --model_type t5 --epochs 2")
    else:
        main()