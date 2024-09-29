# from transformers import AutoTokenizer, AutoModelForCausalLM
# from ctransformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import re
import os

# os.environ["LLAMA_CPP_LIB"] = os.path.join(os.getcwd(), "llama.cpp","libllama.so")
# print(os.path.exists(os.path.join(os.getcwd(), "llama.cpp","libllama.so")))

class vistral_7b:
    def __init__(self, n_gpu_layers) -> None:
        self.model_id = "bartowski/Qwen2.5-3B-Instruct-GGUF"
        self.file_name = "Qwen2.5-3B-Instruct-Q6_K_L.gguf"
        self.n_gpu_layers = n_gpu_layers
        self.model = Llama.from_pretrained(
            repo_id=self.model_id, 
            filename=self.file_name,
            verbose=False,
            n_gpu_layers=self.n_gpu_layers,
        )
    
    def check_advertisement(self, text: str) -> bool:
        prompt = "Check this text is contain advertisement or not. Answer shortly yes or no. The answer is"
        prompt = f"\'{text}\'" + prompt
        answer = self.model(prompt, max_tokens=10,stop=["."])['choices'][0]['text']

        if "yes" in answer.lower():
            return True
        elif "no" in answer.lower():
            return False
        else:
            return None
        
    def check_readability_clarity(self, text: str) -> bool:
        prompt = "Check this text is readable and clear. Answer shortly yes or no. The answer is"
        prompt = f"\'{text}\'" + prompt
        answer = self.model(prompt, max_tokens=10,stop=["."])['choices'][0]['text']

        if "yes" in answer.lower():
            return True
        elif "no" in answer.lower():
            return False
        else:
            return None
    def check_spelling_error(self, text: str) -> bool:
        prompt = "Count the number of Vietnamese spelling errors in this text. There are"
        prompt = f"\'{text}\'" + prompt
        answer = self.model(prompt, max_tokens=10,stop=["."])['choices'][0]['text']
        num = re.findall(r'\d+', answer)

        if len(num) > 0:
            if int(num[0]) >= 5:
                return True
            else:
                return False
        else:
            return None
        
if __name__ == "__main__":
    model = vistral_7b(n_gpu_layers=200)
    # while True:
    #     check = model.check_advertisement("Ngân hàng có app nhẹ nhất, đơn giản và hoài cổ mà tôi tin tưởng")
    #     if check is not None:
    #         break
    # check = model.check_readability_clarity("Ngân hàng có app nhẹ nhất, đơn giản và hoài cổ mà tôi tin tưởng")
    check_advertisement = model.check_advertisement("Ngân hàng có app nhẹ nhất, đơn giản và hoài cổ mà tôi tin tưởng")
    check_readability_clarity = model.check_readability_clarity("Ngân hàng có app nhẹ nhất, đơn giản và hoài cổ mà tôi tin tưởng")
    check_spelling_error = model.check_spelling_error("Ngân hàng có app nhẹ nhất, đơn giản và hoài cổ mà tôi tin tưởng")
    print(check_advertisement, check_readability_clarity, check_spelling_error)