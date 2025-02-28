# Instruction-Tuning Large Language Models (LLMs) with HuggingFace, LoRA and SFTTrainer

This article expalins the Instruction-Tuning method on LLMs using the HuggingFace library with LoRA and SFTTrainer. 

![instruction fine tuning image](https://github.com/user-attachments/assets/3b5350dd-78e3-4fef-9871-75d06ca4c2fd)


Over the past few years, large language models (LLMs) have advanced considerably. Despite their growing capabilities, they often struggle to deliver accurate responses to user inquiries if not optimized well. A variety of methods - prompt engineering, fine-tuning, retrieval-augmented generation (RAG), and systematic prompting - have aimed to close this gap. In this article we will cover Instruction-tuning technique with a code example.

**Objectives:**
- Gain familiarity with a range of template types - including instruction-response, question-answering, summarization, code generation, dialogue, data analysis, and explanation.
- Format datasets accordingly for effective model training. 
- Implement instruction fine-tuning through Hugging Face libraries 
- Utilize Low-Rank Adaptation (LoRA) for efficient LLM optimization. 
- Configure and use the SFTTrainer for supervised fine-tuning.
- Evaluate the model performance with Blue Score.

**What is Instruction Tuning ?**
Instruction tuning (also called Instruction GPT) is a specialized fine-tuning approach where a model is trained on pairs of instructions and outputs, allowing it to learn how to perform specific tasks based on those instructions. With this method you don't have to train your LLM from scratch because you fine-tune a pretrained language model. Instruction tuning usually takes place before performing direct preference optimization (DPO) or reinforcement learning from human feedback (RLHF). We will not cover DPO and PLHF in this article.

![Screenshot 2025-02-26 140215](https://github.com/user-attachments/assets/0ccea478-8fff-4051-aac9-d414499a054a)

In instruction tuning, the dataset is crucial because it presents structured examples of instructions & responses. Let's see some data examples.

--------------------------------------------------------------------------
 
**Instruction and output example:**

Template: `### Instruction: {instruction}\n ### Output: {output}`

Instruction: `Translate the following sentence to German: "Hello, how are you?"`

Output: `Hallo, wie geht's?`

--------------------------------------------------------------------------

**Code generation example:**

Template: `### Task: {task_description}\n ### Code: {code_output}`

Task: Write a function to multiplies two numbers in Python.

Code: `def add(a, b):\n    return a * b`

--------------------------------------------------------------------------

**Data Analysis example:**

Template: `### Instruction: {instruction}\n ### Output: {output}`

Instruction: `Provide insights from the sales data of Q3 2024.`

Output: `The sales decreased by 10% compared to Q2 2024, with the lowest growth in the beauty category.`

--------------------------------------------------------------------------

Let's implement instruction tuning.

#### Install required libraries.

```python

# Intall necessary libraries

!pip install datasets
!pip install trl==0.9.6
!pip install transformers 
!pip install peft 
!pip install tqdm
!pip install numpy 
!pip install pandas
!pip install matplotlib
!pip install seaborn==
!pip install scikit-learn
!pip install sacrebleu
!pip install evaluate
```

#### Import libraries and define device (CPU or GPU).

```python
import io
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import evaluate
import matplotlib.pyplot as plt
from urllib.request import urlopen
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
from trl import SFTConfig 
from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType
import pickle
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

#### Dataset Description

We will use the [CodeAlpaca 20k dataset](https://github.com/sahil280114/codealpaca?tab=readme-ov-file#data-release), a programming-focused data where the instructions asks to write codes. The CodeAlpaca dataset includes the following elements:

`instruction`: String type that describes the specific task for the model. There are 20,000 unique instructions.

`input`: String type that provides additional context for the task. For example, if the instruction is "Write a SQL query that selects all the columns in VBAK table" the input would be the SQL query itself. Approximately 40% of the examples include an input.

`output`: String type that contains the response.

Load the data.

```python
# Load the data
dataset = load_dataset("sahil2801/CodeAlpaca-20k")
# Get only training part
dataset = dataset['train']
# Display and example
dataset[1000]
```

```python
Output:

{'output': 's = "Hello world" \ns = s[::-1] \nprint(s)', 
'instruction': 'Reverse the string given in the input', 
'input': 'Hello world'}
```

We will only focus on data points that have no input to keep it simple and then shuffle.

```python
# Take data points without "input" element
dataset = dataset.filter(lambda example: len(example["input"]) == 0)
# shuffle the data
dataset = dataset.shuffle(seed=31)
```

Split the dataset as train & test.

```python
# Split the dataset
dataset_split = dataset.train_test_split(test_size=0.2, seed=31)
train_dataset = dataset_split['train']
test_dataset = dataset_split['test']

dataset_split
```
```python
Output:

DatasetDict({
    train: Dataset({
        features: ['output', 'instruction', 'input'],
        num_rows: 7811
    })
    test: Dataset({
        features: ['output', 'instruction', 'input'],
        num_rows: 1953
    })
})
```

```python
# Optional: Use a subset of the data to fasten the training time. 
# Since we use CPU, it is better to use a subset to save time.
test_dataset=test_dataset.select(range(1000))
train_dataset=train_dataset.select(range(1000))
```


#### Define the model and tokenizer
We will use the open-source [EleutherAI/gpt-neo-125m · Hugging Face model](https://huggingface.co/EleutherAI/gpt-neo-125m). GPT-Neo 125M model is a transformer model that has 125M parameters created by EleutherAI's replication of the GPT-3. GPT-Neo represents class of models.

```python
# Load the model
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")

# set padding side left
tokenizer.padding_side = 'left'
```

#### Data preprocessing

Before testing the performance of the model, we need to preprocess the test data to generate the prompt. The create_prompt function takes a dataset as input. For each element in the dataset, it structures the instruction and output into a predefined template.

```python
# Preprocess the data and create promts
def create_promt(dataset):
  output_texts_1 = []
  for i in range(len(dataset['instruction'])):
    # create output with response
    text_1 = (
          f"### Instruction:\n{dataset['instruction'][i]}"
          f"\n\n### Output:\n{dataset['output'][i]}</s>"
      )

    output_texts_1.append(text_1)

  return output_texts_1

def create_promt_empty_response(dataset):

  output_texts_2 = []
  for i in range(len(dataset['instruction'])):

    # create output without response
    text_2 = (
        f"### Instruction:\n{dataset['instruction'][i]}"
        f"\n\n### Output:\n"
    )

    output_texts_2.append(text_2)
  return output_texts_2

# tokenize the processed data
expected_outputs = []
instructions_with_responses = create_promt(test_dataset)
only_instructions = create_promt_empty_response(test_dataset)

for instruction_response, instruction in tqdm(zip(instructions_with_responses, only_instructions), total=len(instructions_with_responses)):

    tokenized_response = tokenizer(instruction_response, return_tensors="pt", max_length=1024, truncation=True, padding=False)
    tokenized_instruction = tokenizer(instruction, return_tensors="pt")

    # Extract the expected output by decoding the difference
    expected_output = tokenizer.decode(
        tokenized_response['input_ids'][0][len(tokenized_instruction['input_ids'][0])-1:],
        skip_special_tokens=True
    )

    expected_outputs.append(expected_output)

```

Print the processed results

```python
print('instructions\n' + only_instructions[0], '\n')
print(' instructions_with_responses\n' + instructions_with_responses[0], '\n')
print('expected_outputs' + expected_outputs[0])
```
```python
Output:

instructions
### Instruction:
Name the most important benefit of using a database system.
### Response:

instructions_with_responses
### Instruction:
Name the most important benefit of using a database system.
### Response:
The most important benefit of using a database system is the ability to store and retrieve data quickly and easily. Database systems also provide support for data security, data integrity, and concurrently accessing and modifying data from multiple systems.</s> 

expected_outputs
The most important benefit of using a database system is the ability to store and retrieve data quickly and easily. Database systems also provide support for data security, data integrity, and concurrently accessing and modifying data from multiple systems.
```

It's advantageous to transform into a torch ListDataset. The code below introduces a class named ListDataset, which converts a list into a torch-compatible dataset.

```python
# Define the ListDataset object
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    
    def __len__(self):
        return len(self.original_list)
    
    def __getitem__(self, i):
        return self.original_list[i]
instructions_torch = ListDataset(only_instructions)
```

#### Implement instruction fine-tuning with LoRA

Fine-tuning the model for all the parameters could be time consuming and waste of resources. To optimize time, we'll use a parameter-efficient fine-tuning (PEFT) approach called [Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/diffusers/en/training/lora) for instruction fine-tuning. First, transform the model into a PEFT-compatible version for LoRA by defining a LoraConfig object from the peft library, specifying parameters such as the LoRA rank and target modules. Then, apply the LoRA configuration to the model using `get_peft_model()`, which effectively converts it into a LoRA model.

```python
#Parameters:
#r: low-rank dimension
#lora_alpha: scaling factor of LoRa
#target_modules: Modules to apply LoRA
#lora_dropout: Dropout rate
#task_type: #task type which is causal language model

lora_model_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)

# convert our model into a lora model
model = get_peft_model(model, lora_model_config)
```

Next we need to mask the generated responses, because we will be using the [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) which outputs the instructions and the responses together. To evaluate the quality of the generated output, we can only consider the response text and not instruction. Therefore, we will mask the instruction part of the output text and only keep the response part of the output. We will use the `DataCollatorForCompletionOnlyLM` class from `trl` library. 

```python
# Define response template
response_template = "### Output:\n"
# Define function that masks
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
```

Using the masking in DataCollatorForCompletionOnlyLM we can exclude the instruction part from the loss calculation.

#### Training
For the training we first define the SFTConfig, and then define the `SFTTrainer` object. The `SFTConfig` creates the parameters of the `SFTTrainer` such as number of epochs, batch size, max sequence length, etc. We then feed the `SFTConfig` createsto give training parameters to SFTTrainer.

```python
# Define configuration and training parameters
training_args = SFTConfig(
    output_dir="/tmp",
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=True,
    per_device_train_batch_size=2,  # Reduce batch size
    per_device_eval_batch_size=2,  # Reduce batch size
    max_seq_length=1024,
    do_eval=True
)

# Define SFTTrainer object
trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    formatting_func=create_promt,
    args=training_args,
    packing=False,
    data_collator=collator,
)
```

Assign the end of token (EOS) to the tokenizer.

```python
#Assign the enf of token (EOS) to the tokenizer.
tokenizer.pad_token = tokenizer.eos_token
```


Start training, save the training history to save the loss, and lastly save the model for later usage. 

**PS:** We only train 3 epochs since I'm using CPU. The training can be too long if you use CPU.

```python
# Start training
trainer.train()
# Save the history
log_history_lora = trainer.state.log_history
# Save the model
trainer.save_model("./final_model_instruction_tuning_with_lora")
```

**Note:** To train the model you should open a free account at [Weights & Biases: The AI Developer Platform (wandb.ai)](https://wandb.ai/home) and get a free key.

We can plot the loss during training.

```python
# Extract the training loss
train_loss = [log["loss"] for log in log_history_lora if "loss" in log]

# Plot the training loss
plt.figure(figsize=(6, 3))
plt.plot(train_loss, label='Training Loss')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
```
The below image shows the training loss through epochs.

![download](https://github.com/user-attachments/assets/5d03415f-f3e0-4379-900e-3685e399447e)


Here it is important to note that we only trained the model 3 epochs because of the insufficient resources (CPU instead of GPU).

After training we define the text generation pipeline to test the model performance.

```python
# Define the text generation pipeline
generation_pipeline = pipeline("text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        device=device, 
                        batch_size=2, 
                        max_length=50, 
                        truncation=True, 
                        padding=False,
                        return_full_text=False)

```

#### Generate responses for the test data

Now we can generate the responses using the generation_pipeline which generates tokens trained model. Note that it takes so much time to generate responses if you use CPU.

```python
# generate responses
with torch.no_grad():
    pipeline_iterator= generation_pipeline(instructions_torch,
                                #max length of responses
                                max_length=50, 
                                num_beams=5,
                                early_stopping=True,)
# Save the generated outputs
generated_outputs = []
for text in pipeline_iterator:
    generated_outputs.append(text[0]["generated_text"])
```

Print example response.

```python
print(instructions[0])
print(expected_outputs[0])
print(generated_outputs_lora[0])
```

```python
Output:

### Instruction:
Define a Bash function with name 'times2' which takes one parameter and prints the value doubled.

### Output:
 


times2 () {
  value=$1
  echo "$value*2 = $(($value*2))"
}</s> 

def times2(name):
    print(name)
</s>
```

#### Test the performance with Blue Score

We can evaluate the alignments between the model's generated responses and the expected responses using BLEU score, a metric originally designed to assess the quality of machine translation outputs. BLEU calculates a similarity score by comparing each generated segment with a set of reference outputs, and the overall score is obtained by averaging these individual comparisons. Depending on the implementation, BLEU scores range either bettwen [0,1] or between [0,100]. Higher values indicate a stronger match between the generated and expected outputs.

BLEU scores may be a difficult evaluation method due to various implementation parameters. To address this, we use [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu), a standardized variant of BLEU that ensures consistency across different evaluations.

```python
import evaluate
import sacrebleu

# Load the Sacreblue metric
sacrebleu = evaluate.load("sacrebleu")
results_base = sacrebleu.compute(predictions=generated_outputs,
                                 references=expected_outputs[:10])
```

```python
print(list(results_base.keys()))
print(round(results_base["score"], 1))
```

```python
Output:

['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
0.4
```

You can see that the fine-tuned model achieves a SacreBLEU score of 0.4. The reason why the score is too low is that we trained the model for 3 epochs only since we did run it on CPU. Therefore, the model couldn't learn and generalize well. You can use the code above to implement the instruct tuning method on LLMs.


