import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from torch.nn.functional import cosine_similarity

class AdversarialModel:
    def __init__(self, model_name="TheBloke/vicuna-7B-v1.5-GPTQ"):
        """
        Initializes the AdversarialModel with a given model name.
        Args:
        - model_name (str): Name of the model to use.
        """
        common_config = {
            'model_name_or_path': model_name,
            'model_basename': "model",
            'use_safetensors': True,
            'trust_remote_code': True
        }

        self.model = AutoGPTQForCausalLM.from_quantized(
            **common_config,
            use_fast=True,
            device='cuda:0',
            low_cpu_mem_usage=True,
            use_cache=False,
            use_triton=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            padding_side="left",
            **common_config
        )
        self.model.eval().to('cuda:0')
        self.tokenizer.pad_token = "PAD"
        self.model.config.pad_token_id = "PAD"

    def match_tensor_lengths(self, tensor_a, tensor_b):
        """
        Pads or truncates tensor_b to match the sequence length of tensor_a.
        """
        seq_len_a = tensor_a.shape[1]
        seq_len_b = tensor_b.shape[1]

        if seq_len_b < seq_len_a:
            padding = torch.zeros((tensor_b.shape[0], seq_len_a - seq_len_b, tensor_b.shape[2]), device=tensor_b.device)
            tensor_b = torch.cat([tensor_b, padding], dim=1)
        elif seq_len_b > seq_len_a:
            tensor_b = tensor_b[:, :seq_len_a, :]

        return tensor_b


    def adversarial_attack(self, raw_text, desired_text, max_iterations=10, epsilon=0.01):
        """
        Generates adversarial text based on the given input text and desired text.
        Args:
        - raw_text (str): The original text to be attacked.
        - desired_text (str): The desired adversarial text.
        - max_iterations (int): Maximum number of iterations for the attack.
        - epsilon (float): The step size for perturbations.
        Returns:
        - perturbed_text (str): The adversarially perturbed text.
        """
        self.model.train()

        inputs = self.tokenizer(raw_text, return_tensors="pt", truncation=True, padding=True).to('cuda:0')
        embeddings = self.model.get_input_embeddings()(inputs["input_ids"])

        desired_inputs = self.tokenizer(desired_text, return_tensors="pt", truncation=True, padding=True).to('cuda:0')
        desired_embeddings = self.model.get_input_embeddings()(desired_inputs["input_ids"])

        for _ in range(max_iterations):
            embeddings.requires_grad_(True)
            outputs = self.model(inputs_embeds=embeddings)
            desired_embeddings_matched = self.match_tensor_lengths(outputs.logits, desired_embeddings)
            similarity = cosine_similarity(outputs.logits.squeeze(0), desired_embeddings_matched.squeeze(0)).mean()

            loss = -similarity
            loss.backward()

            perturbed_embeddings = embeddings - epsilon * torch.sign(embeddings.grad).detach()
            perturbed_input_ids = torch.argmax(self.model.get_output_embeddings()(perturbed_embeddings), dim=2)

            perturbed_text = self.tokenizer.decode(perturbed_input_ids[0], skip_special_tokens=True)

        self.model.eval()
        return perturbed_text

# Example usage
adversary = AdversarialModel()
raw_text = "The movie was great!"
desired_text = "This movie was bad!"
adversarial_result = adversary.adversarial_attack(raw_text, desired_text)
print(adversarial_result)
