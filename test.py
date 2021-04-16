import torch
import transformers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertModel, BertTokenizerFast


def valid_tokenizing(sent_batch, tokenizer, max_length = 35):
    
    markers = ["[CLS]", "[SEP]"]

    valid_indices = []
    input_ids = []
    attention_mask = []

    for sent in sent_batch:
        tokenized_sequence = tokenizer.encode(" ".join(sent))
        valid_idx = []

        input_ids.append(torch.LongTensor(tokenized_sequence.ids))
        attention_mask.append(torch.LongTensor(tokenized_sequence.attention_mask))

        for token in tokenized_sequence.tokens:
            if token in markers or re.search("^##.*$", token):
                valid_idx.append(0)
            else:
                valid_idx.append(1)
        
        valid_indices.append(torch.LongTensor(valid_idx))

    valid_indices = pad_sequence(valid_indices, batch_first=True, padding_value=0.0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0.0)
    token_type_ids = torch.zeros(input_ids.shape, dtype = torch.long)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

    bert_input = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}

    return bert_input, valid_indices
#tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

# text_1 = "Who was Jim Henson ?"

# # Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
# indexed_tokens = tokenizer.encode(text_1, add_special_tokens=True)
# print(indexed_tokens)

# # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0]*len(indexed_tokens)

# # Convert inputs to PyTorch tensors
# segments_tensors = torch.tensor([segments_ids])
# tokens_tensor = torch.tensor([indexed_tokens])

##model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

# with torch.no_grad():
#     encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)

# print(encoded_layers)


batch_sentences = ["hello, i'm Testing this efauenufefu","hello, i'm this efauenufefu"]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer2 = BertTokenizerFast("bert-base-cased-vocab.txt", wordpieces_prefix = "##")
model = BertModel.from_pretrained('bert-base-cased')
tokenizer3 = BertWordPieceTokenizer("bert-base-cased-vocab.txt")
tokenized_sequence = tokenizer3.encode("hello, i'm te~sting this efauenufefu")

inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length = 15)#, padding=True,truncation=True, max_length = 10 )

for i in range(len(batch_sentences)):
    #decoded = tokenized_sequence.decode(inputs["input_ids"][i],clean_up_tokenization_spaces = False)
    print(tokenized_sequence)

input2 = {'input_ids':torch.LongTensor(tokenized_sequence.ids).view(1, -1), 'token_type_ids':torch.LongTensor(tokenized_sequence.type_ids).view(1, -1), 'attention_mask':torch.LongTensor(tokenized_sequence.attention_mask).view(1, -1) }


outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state