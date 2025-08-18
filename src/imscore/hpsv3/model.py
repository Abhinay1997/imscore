
import torch
import torch.nn as nn
import torchvision.transforms as T
import huggingface_hub
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from .utils import process, INSTRUCTION


class HPSv3(Qwen2VLForConditionalGeneration):
    def __init__(self, config, special_token_ids=None):
        super().__init__(config)
        self.rm_head = nn.Sequential(
            nn.Linear(3584, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
        self.rm_head.to(torch.float32)
        self.special_token_ids = special_token_ids
        self.reward_token = "special"
        self.rm_head.to(torch.float32)

        

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None= None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
    ):
        inputs_embeds = self.model.embed_tokens(input_ids)
        pixel_values = pixel_values.type(self.visual.get_dtype())
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(input_ids=None, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states)

        hidden_states = outputs[0]  # [B, L, D]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            logits = self.rm_head(hidden_states)  # [B, L, N]

        b, *_ = inputs_embeds.shape
        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in self.special_token_ids:
            special_token_mask = special_token_mask | (input_ids == special_token_id)
        pooled = logits[special_token_mask, ...]
        pooled = pooled.view(b, 1, -1)  # [B, 3, N] assert 3 attributes
        pooled = pooled.view(b, -1)


        return { "logits": pooled }

class HPSv3Reward():
    def __init__(self, device='cuda'):
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side="right")
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)
        model = HPSv3.from_pretrained("RE-N-Y/hpsv3", special_token_ids=special_token_ids, torch_dtype=torch.bfloat16)
        model.to(torch.bfloat16)
        model.rm_head.to(torch.float32)
        model.config.tokenizer_padding_side = processor.tokenizer.padding_side

        self.device = device
        self.use_special_tokens = True
        self.model = model
        self.processor = processor
        self.model.to(self.device)
    
    def prepare(self, images, prompts):
        messages = []
        tform = T.ToPILImage()
        images = [tform(im) for im in images]
        for text, image in zip(prompts, images):
            m = [{ 
                "role": "user",
                "content": [ 
                    { "type": "image", "image": image, "minpix": 256 * 28 * 28, "maxpix" : 256 * 28 * 28 },
                    { "type": "text",  "text": INSTRUCTION.format(prompt=text) }
                ]
            }]

            messages.append(m)

        images = process(messages)
        batch = self.processor(
            text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=images,
            padding=True,
            return_tensors="pt",
        )
        batch = { k : v.to(self.device) for k,v in batch.items() }
        return batch

    def score(self, images, prompts):
        
        batch = self.prepare(images, prompts)
        rewards = self.model(**batch)["logits"]

        return rewards