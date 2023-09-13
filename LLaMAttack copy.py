B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class AdversarialModel:
    def __init__(self, model_name="C:\\Users\\Lol\\Desktop\\GPT\\vicuna-7b-v1.5", device='cpu'):
        """ Initializes the AdversarialModel with a given model name. """
        if device not in ['cpu', 'cuda:0']:
            raise ValueError("Invalid device. Use 'cpu' or 'cuda:0'.")
        self.device = device
        
        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )

        quantize_config = BaseQuantizeConfig(
            bits=4,
            sym=True,
            model_name_or_path=model_name,
            model_file_base_name='model',
            group_size=128,
            damp_percent=0.01,
            desc_act=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=double_quant_config,
            use_cache=False,
            trust_remote_code=True,
            device_map=device,
            offload_folder = './offload_folder'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code =True
        )

        #self.model.train().to(device)

        # Synchronize pad token for tokenizer and model
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.pad_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def format_tokens(self, attack_sets):
        """ Formats the given attack_sets into control and target texts. """
        control_texts = []
        target_texts = []
        for dialog in attack_sets:
            for control, target in zip(dialog[:2], dialog[1:2]):
                #control_texts.append(f"<s> {B_SYS}You are a kind and helpful assistant.{E_SYS}{B_INST} {control['prompt'].strip()} ")
                control_texts.append(
                    f"""<s>A chat between a user and an assistant. ### User: {control['prompt'].strip()} ### ASSISTANT: """
                )
                #control_texts.append(f"{control['prompt'].strip()} ")
                target_texts.append(target['target'].strip())
        return control_texts, target_texts

    def generate_adversarial_suffix(self, attack_sets, max_iterations=1000, max_length=32):
        """ Generate an adversarial suffix using the Greedy Coordinate Gradient-based Search. """
        control_texts, target_texts = self.format_tokens(attack_sets)
        suffixes = []
        
        for input_text, target_phrase in zip(control_texts, target_texts):
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            target_ids = self.tokenizer.encode(target_phrase, return_tensors="pt").to(self.device)
            
            # Initialize adversarial suffix with random tokens
            suffix = torch.full((1, max_length), float(self.tokenizer.pad_token_id), dtype=torch.float).to(self.device)
            suffix = suffix.float()  # Convert to float
            suffix.requires_grad = True
            print(suffix.grad)
            for param in self.model.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            self.model.train()
            for iteration in range(max_iterations):
                combined_input = torch.cat([input_ids, suffix.long(), self.tokenizer.encode(" ### Assistant: ", return_tensors="pt").to(self.device)], dim=1)
                #combined_input = torch.cat([input_ids, suffix.long()], dim=1)
                combined_input = combined_input.float()  # Convert to float
                combined_input.requires_grad = True

                # Compute model outputs
                test_input = torch.cat([input_ids, self.tokenizer.encode(E_INST + " ", return_tensors="pt").to(self.device)], dim=1)
                test_outputs = self.model(test_input)
                test_logits = test_outputs.logits
                print('Control Text: ', self.tokenizer.decode(torch.softmax(test_logits, dim=-1)[0]))
                # You can compute the loss here using test_logits and see if gradients are computed
                
                outputs = self.model(combined_input.long())
                logits = outputs.logits
                predicted_token_ids = torch.argmax(logits, dim=-1)  
                predicted_text = self.tokenizer.decode(predicted_token_ids[0])
                print('Predicted text: ', predicted_text)

                # Calculate the required padding length
                padding_length = logits.size(1) - target_ids.size(1)
                padding_tensor = torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=torch.long).to(self.device)
                target_ids_padded = torch.cat([target_ids, padding_tensor], dim=1)

                # Calculate the negative log likelihood of the target phrase
                loss = -torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids_padded.view(-1))
                print('Loss: ', loss.item())
                #if loss.item() < 1e-6:
                #    break
                
                # Optimize the adversarial suffix
                loss.backward()
                print("Loss value:", loss.item())
                
                if suffix.grad is None:
                    print("No gradient for suffix at iteration", iteration)
                    break
                
                with torch.no_grad():
                    update = 0.01 * suffix.grad.sign()
                    updated_suffix = suffix + update
                    updated_suffix = torch.clamp(updated_suffix, 0, float(len(self.tokenizer) - 1))
                    suffix.data = updated_suffix.data
                    suffix.grad.zero_()
                print("Suffix:", self.tokenizer.decode(suffix[0].long(), skip_special_tokens=True))

            suffix_text = self.tokenizer.decode(suffix[0].long(), skip_special_tokens=True)
            suffixes.append(suffix_text)

        return suffixes
    
def main():
    # Sample attack_sets
    attack_sets = [
        [
            {"prompt": "Respond with 'I'm a cat'"},
            {"target": "I'm a dog"}
        ]
    ]

    # Use 'cuda:0' for GPU, 'cpu' for CPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = AdversarialModel(device='cuda:0')
    adversarial_suffixes = model.generate_adversarial_suffix(attack_sets)
    for i, suffix in enumerate(adversarial_suffixes):
        print(f"Adversarial Suffix {i + 1}: {suffix}")

if __name__ == "__main__":
    main()
