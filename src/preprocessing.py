from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Keyurjotaniya007/wmt-hi-en-tokenizer")

src_max_length = 32
tgt_max_length = 32

bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

def reversed_source_tokens(input_ids, bos_token_id=None, eos_token_id=None):
    if len(input_ids) <= 2:
        return input_ids

    if bos_token_id is not None and eos_token_id is not None:
        if input_ids[0] == bos_token_id and input_ids[-1] == eos_token_id:
            middle = input_ids[1:-1]
            return [bos_token_id] + middle[::-1] + [eos_token_id]

    return input_ids[::-1]

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["hi"] for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=src_max_length
    )

    labels = tokenizer(
        text_target=targets,
        truncation=True,
        max_length=tgt_max_length
    )

    reversed_input_ids = []
    for ids in model_inputs["input_ids"]:
        reversed_ids = reversed_source_tokens(
            ids,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        reversed_input_ids.append(reversed_ids)

    model_inputs["input_ids"] = reversed_input_ids
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

test_tokenized = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=test_dataset.column_names
)
