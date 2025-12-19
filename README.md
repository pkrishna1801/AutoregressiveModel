# Grammar-to-Natural Language Model

A PyTorch-based training pipeline for converting grammatical annotations to natural language sentences. Supports multiple model architectures including T5, GPT-2, and Llama with LoRA fine-tuning.

## Features

- **Multiple Model Support**: Train T5 (seq2seq), GPT-2 (causal), or Llama-2 with LoRA
- **CSV Data Pipeline**: Easy data loading from CSV files with automatic train/val/test splitting
- **Grammar Pattern Analysis**: Built-in tools to analyze grammar tag distributions
- **Comprehensive Evaluation**: BLEU and ROUGE metrics for model evaluation
- **Mixed Precision Training**: Automatic FP16 training when GPU is available
- **Flexible Configuration**: Customizable hyperparameters via command line arguments

## Installation

```bash
pip install torch transformers pandas numpy scikit-learn nltk rouge peft bitsandbytes
```

For Llama-LoRA training, ensure you have access to the Llama model weights from Hugging Face.

## Data Format

Your CSV file should have at least two columns for input (grammar annotations) and output (natural sentences):

| input | output |
|-------|--------|
| `[subj]Cat[verb]eat[obj]mouse[tense]past` | The cat ate the mouse. |
| `[subj]Dog[verb]run[location]park[tense]present` | The dog runs in the park. |

### Supported Grammar Tags

The model recognizes tags such as:
- `[subj]` - Subject
- `[verb]` - Verb
- `[obj]` - Object
- `[location]` - Location
- `[time]` - Time expression
- `[tense]` - Tense marker
- `[adjective]`, `[adverb]`, `[pronoun]`, `[preposition]`

## Usage

### Basic Training

```bash
python SVO-AuroRegressive\ Model.py \
    --data_csv data.csv \
    --model_type t5 \
    --epochs 10 \
    --batch_size 8 \
    --output_dir ./model_output
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_csv` | *required* | Path to CSV file with training data |
| `--model_type` | `t5` | Model type: `t5`, `gpt`, or `llama-lora` |
| `--model_name` | Auto | Base model name (e.g., `t5-small`, `gpt2`) |
| `--output_dir` | `./model_output` | Output directory for saved models |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `8` | Training batch size |
| `--learning_rate` | `3e-4` | Learning rate |
| `--input_column` | `input` | CSV column name for grammar input |
| `--output_column` | `output` | CSV column name for natural output |
| `--test_size` | `0.1` | Test set proportion |
| `--val_size` | `0.1` | Validation set proportion |

### Training Different Models

**T5 (Recommended for beginners):**
```bash
python SVO-AuroRegressive\ Model.py --data_csv data.csv --model_type t5 --model_name t5-base
```

**GPT-2:**
```bash
python SVO-AuroRegressive\ Model.py --data_csv data.csv --model_type gpt --model_name gpt2-medium
```

**Llama-2 with LoRA (requires GPU):**
```bash
python SVO-AuroRegressive\ Model.py --data_csv data.csv --model_type llama-lora --batch_size 2
```

## Programmatic Usage

### Data Preprocessing

```python
from train import DataPreprocessor

# Validate your CSV
DataPreprocessor.validate_csv('data.csv', input_col='input', output_col='output')

# Analyze grammar patterns
tag_counts = DataPreprocessor.analyze_grammar_patterns('data.csv', input_col='input')

# Split into train/val/test
train_df, val_df, test_df = DataPreprocessor.split_csv_data(
    'data.csv',
    output_dir='./splits',
    test_size=0.1,
    val_size=0.1
)
```

### Inference

```python
from train import GrammarToNaturalInference

# Load trained model
inference = GrammarToNaturalInference(
    model_path='./model_output/final_model',
    model_type='t5'
)

# Single prediction
result = inference.generate('[subj]Bird[verb]fly[location]sky[tense]present')
print(result)  # "The bird flies in the sky."

# Batch prediction
inputs = [
    '[subj]Cat[verb]sleep[location]sofa[tense]present',
    '[subj]Children[verb]play[location]garden[tense]past'
]
results = inference.batch_generate(inputs)
```

### Custom Training

```python
from train import train_t5_model, train_gpt_model

# Train T5 model
model, tokenizer = train_t5_model(
    train_csv_path='train.csv',
    val_csv_path='val.csv',
    model_name='t5-small',
    output_dir='./my_model',
    num_epochs=10,
    batch_size=8,
    learning_rate=3e-4
)
```

## Evaluation Metrics

The pipeline evaluates models using:

- **BLEU Score**: Measures n-gram precision against reference translations
- **ROUGE-1 F1**: Unigram overlap
- **ROUGE-2 F1**: Bigram overlap
- **ROUGE-L F1**: Longest common subsequence

### Example Results (T5-small, 10 epochs)

| Metric | Score |
|--------|-------|
| BLEU | 0.6004 |
| ROUGE-1 F1 | 0.7711 |
| ROUGE-2 F1 | 0.6798 |
| ROUGE-L F1 | 0.7699 |

## Output Structure

```
model_output/
├── train.csv              # Training split
├── val.csv                # Validation split
├── test.csv               # Test split
├── logs/                  # TensorBoard logs
├── checkpoint-*/          # Training checkpoints
└── final_model/           # Final saved model
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── ...
```

## Tips for Better Results

1. **More Data**: Larger datasets generally improve performance
2. **Longer Training**: Try 15-20 epochs for complex grammar patterns
3. **Larger Models**: Use `t5-base` or `t5-large` for better quality
4. **Consistent Formatting**: Ensure grammar tags are consistently formatted
5. **Balanced Dataset**: Include diverse sentence structures and tenses

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `--batch_size` (try 4 or 2)
- Use gradient accumulation (already set to 2 for GPT)
- Use `--model_type llama-lora` for memory-efficient fine-tuning

**Poor Generation Quality:**
- Increase training epochs
- Use a larger base model
- Check data quality and consistency

**Slow Training:**
- Ensure CUDA is available (`torch.cuda.is_available()`)
- Enable mixed precision (automatic when GPU detected)

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{grammar_to_natural,
  title={Grammar-to-Natural Language Model},
  year={2024},
  url={https://github.com/yourusername/grammar-to-natural}
}
```
