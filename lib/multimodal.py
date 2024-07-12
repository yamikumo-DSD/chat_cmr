import os
import PIL
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports


class Florence2Large:
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