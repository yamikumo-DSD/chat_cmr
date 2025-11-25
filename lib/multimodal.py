import os
import PIL
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports
import torch
from abc import ABC, abstractmethod



def shrink_image(image_path: str, length: int) -> str:
    """
    Shrink the *shorter* edge (width or height) to 640px,
    keep aspect ratio, and save as <uuid>.png where
    uuid = hash of image_path.

    Returns the output file path.
    """
    import os
    import uuid
    from PIL import Image

    # Make a deterministic UUID based on the image path
    img_uuid = uuid.uuid5(uuid.NAMESPACE_URL, image_path)
    out_filename = f"{img_uuid}.png"
    out_dir = os.path.dirname(image_path) or "."
    out_path = os.path.join(out_dir, out_filename)

    # Open and resize
    with Image.open(image_path) as im:
        w, h = im.size
        short_edge = min(w, h)

        # If already smaller than or equal to 640 on the short side,
        # don't upscale, just save as PNG with the new name
        if short_edge <= length:
            im.convert("RGB").save(out_path, format="PNG")
            return out_path

        scale = length / short_edge
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized.convert("RGB").save(out_path, format="PNG")

    return out_path



class Captioner(ABC):
    @abstractmethod
    def get_caption(self, image, *args, **kwargs) -> str:
        pass


class Florence2Large(Captioner):
    _device: str
    _model: AutoModelForCausalLM
    _processor: AutoProcessor

    @staticmethod
    def _fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
        
        if not str(filename).endswith("/modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        try:
            imports.remove("flash_attn")
        except:
            ...
        return imports

    def load_model(self, hf_repo: str = "microsoft/Florence-2-large-ft") -> None:
        from unittest.mock import patch
        
        with patch("transformers.dynamic_module_utils.get_imports", Florence2Large._fixed_get_imports):
            self._model = AutoModelForCausalLM.from_pretrained(hf_repo, trust_remote_code=True)
            self._model.to(self._device)
            self._processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)

    def __init__(self, use_accelerator: bool = True) -> None:
        from lib.backends import suggest_device
        self._device = suggest_device() if use_accelerator else "cpu"
        self.load_model()

    def get_caption(
        self, 
        image,
        prompt: str = "<MORE_DETAILED_CAPTION>"
    ) -> str:
        """
        Args:
            image: Pillow image object, or path or url to an image.
        """
        def is_url(path: str) -> bool:
            import re
            return bool(re.search(r"^https?://", path))
        
        if isinstance(image, str):
            if is_url(image):
                import requests
                image = PIL.Image.open(requests.get(image, stream=True).raw)
            else:
                image = PIL.Image.open(image)
        
        image = image.convert("RGB")
        
        inputs = self._processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(self._device)
        
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
        parsed_answer = self._processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
    
        return parsed_answer[prompt]



class Florence2BasePromptGen(Captioner):
    _device: str
    _model: AutoModelForCausalLM
    _processor: AutoProcessor

    @staticmethod
    def _fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
        
        if not str(filename).endswith("/modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports

    def load_model(self, hf_repo: str = "MiaoshouAI/Florence-2-base-PromptGen") -> None:
        from unittest.mock import patch
        
        with patch("transformers.dynamic_module_utils.get_imports", Florence2Large._fixed_get_imports):
            self._model = AutoModelForCausalLM.from_pretrained(hf_repo, trust_remote_code=True)
            self._model.to(self._device)
            self._processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)

    def __init__(self, use_accelerator: bool = True) -> None:
        from lib.backends import suggest_device
        self._device = suggest_device() if use_accelerator else "cpu"
        self.load_model()

    def get_caption(
        self, 
        image,
        prompt: str = "<GENERATE_PROMPT>"
    ) -> str:
        """
        Args:
            image: Pillow image object, or path or url to an image.
        """
        def is_url(path: str) -> bool:
            import re
            return bool(re.search(r"^https?://", path))
        
        if isinstance(image, str):
            if is_url(image):
                import requests
                image = PIL.Image.open(requests.get(image, stream=True).raw)
            else:
                image = PIL.Image.open(image)
        
        image = image.convert("RGB")
        
        inputs = self._processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(self._device)
        
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
        parsed_answer = self._processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
    
        return parsed_answer[prompt]



class HuihuiQwen3VLNSFWQA(Captioner):
    """
    Qwen3-VL has ability as a visual agent.
    Qwen3-VL's coordinate system is relative coordinate (x,y)=(0~999,0~999) regardless of the original image shape.
    (0,0)___________
        |          |
        |          |
        |          |
        |__________|
                 (999,999)
    """
    def load_model(self, hf_repo: str = "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated") -> None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_repo, dtype=torch.float16,
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(hf_repo)

    def __init__(self, use_accelerator: bool = True) -> None:
        from lib.backends import suggest_device
        self._device = suggest_device() if use_accelerator else "cpu"
        self.load_model()

    def get_caption(
        self, 
        image,
        prompt: str = "Describe the image.",
        max_new_tokens: int = 512,
        shrink: int|None = 640,
    ) -> str:
        from lib.utils import is_valid_path
        import PIL
        import os
        
        try:
            import os
            if is_valid_path(image)["is_valid"]:
                pass
            else: # In the case of PIL.Image object.
                temp_file_name = "huihui_temp_file.png"
                image.convert("RGB").save(temp_file_name, format="PNG")
                image = temp_file_name

            if shrink is not None:
                image = shrink_image(image, 640)
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self._device)
            
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            #os.remove(image)
        except BaseException as e:
            return f"An error occured while captioning ({str(e)})."
    
        return output_text