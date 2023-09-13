import torch
from transformers import AutoTokenizer, LlamaModel
from transformers.models.llama import modeling_llama

class AdversarialModel:
    def __init__(self, model_name="C:\\Users\\Lol\\Desktop\\GPT\\vicuna-7b-v1.5", device='cpu'):
        self._validate_device(device)
        self.device = device
        self.model = self._load_model(model_name)
        self.tokenizer = self._load_tokenizer(model_name)

    @staticmethod
    def _validate_device(device):
        if device not in ['cpu', 'cuda:0']:
            raise ValueError("Invalid device. Use 'cpu' or 'cuda:0'.")
    
    def _load_model(self, model_name):
        return LlamaModel.from_pretrained(model_name, trust_remote_code=True, device_map='cpu')
    
    def _load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        return tokenizer

    def format_tokens(self, attack_sets):
        control_texts, target_texts = [], []
        for dialog in attack_sets:
            for control, target in zip(dialog[:1], dialog[1:]):
                user_prompt = control['prompt'].strip()
                control_texts.append(f"<s> A chat between a user and an assistant. ### User: {user_prompt} ### ASSISTANT: ")
                control_texts.append(user_prompt)
                target_texts.append(target['target'].strip())
        return control_texts, target_texts

    def _load_layers(self, start_layer, end_layer):
        return [
            self._load_layer(i) for i in range(start_layer, end_layer + 1)
        ]

    def _load_layer(self, layer_number):
        layer_path = f"offload_folder/layer_{layer_number-1}.pt"
        layer = modeling_llama.LlamaDecoderLayer(self.model.config).to(self.device)
        layer.load_state_dict(torch.load(layer_path))
        return layer

    def process_layers_in_chunks(self, input_data, total_layers=32, chunk_size=5, attention_mask=None, position_ids=None):
        input_ids = self.tokenizer.encode(input_data, return_tensors="pt").to(self.device)
        final_output = input_ids
        
        for start_layer in range(1, total_layers + 1, chunk_size):
            end_layer = min(start_layer + chunk_size - 1, total_layers)
            layers = self._load_layers(start_layer, end_layer)
            
            for layer in layers:
                layer_output = layer(final_output, attention_mask=attention_mask, position_ids=position_ids, use_cache=True, output_attentions=True)
                final_output = layer_output[0]
                del layer  # Free up memory

            if self.device == 'cuda:0':
                torch.cuda.empty_cache()

        return final_output

    def generate_adversarial_suffix(self, attack_sets, max_iterations=1000, max_length=32):
        control_texts, target_texts = self.format_tokens(attack_sets)
        return [
            self._find_suffix(control_text, target_text, max_iterations, max_length)
            for control_text, target_text in zip(control_texts, target_texts)
        ]

    def _find_suffix(self, input_text, target_phrase, max_iterations, max_length):
        input_ids = self.tk.encode(input_text, return_tensors="pt").to(self.dev)
        target_ids = self.tk.encode(target_phrase, return_tensors="pt").to(self.dev)
        suffix = self._initialize_suffix(max_length)

        for iteration in range(max_iterations):
            comb_inp = self._combine_input(input_ids, suffix)
            test_input = torch.cat([input_ids, self.tk.encode(" ### Assistant: ", return_tensors="pt").to(self.dev)], dim=1)
            test_outputs = self.mdl(test_input)
            test_logits = test_outputs.logits
            print('Control Text: ', self.tk.decode(torch.softmax(test_logits, dim=-1)[0]))
            outputs = self.mdl(comb_inp.long())
            logits = outputs.logits
            loss = self._calculate_loss(logits, target_ids)
            print("Loss value:", loss.item())

            if suffix.grad is None:
                print("No gradient for suffix at iteration", iteration)
                break

            self._update_suffix(suffix)

        return self.tk.decode(suffix[0].long(), skip_special_tokens=True)

    def _initialize_suffix(self, max_length):
        suffix = torch.full((1, max_length), float(self.tk.pad_token_id), dtype=torch.float).to(self.dev)
        suffix = suffix.float()
        suffix.requires_grad = True

        for param in self.mdl.parameters():
            if param.dtype.is_floating_point:
                param.requires_grad = True

        self.mdl.train()
        return suffix

    def _combine_input(self, input_ids, suffix):
        return torch.cat([input_ids, suffix.long(), self.tk.encode(" ### Assistant: ", return_tensors="pt").to(self.dev)], dim=1).float()

    def _calculate_loss(self, logits, target_ids):
        padding_length = logits.size(1) - target_ids.size(1)
        padding_tensor = torch.full((1, padding_length), self.tk.pad_token_id, dtype=torch.long).to(self.dev)
        target_ids_padded = torch.cat([target_ids, padding_tensor], dim=1)
        loss = -torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids_padded.view(-1))
        loss.backward()
        return loss

    def _update_suffix(self, suffix):
        with torch.no_grad():
            update = 0.01 * suffix.grad.sign()
            updated_suffix = suffix + update
            updated_suffix = torch.clamp(updated_suffix, 0, float(len(self.tk) - 1))
            suffix.data = updated_suffix.data
            suffix.grad.zero_()

def main():
    attack_sets = [
        [
            {"prompt": "Respond with 'I'm a cat'"},
            {"target": "I'm a dog"}
        ]
    ]
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = AdversarialModel(device=dev)
    model.process_layers_in_chunks("test")

if __name__ == "__main__":
    main()
