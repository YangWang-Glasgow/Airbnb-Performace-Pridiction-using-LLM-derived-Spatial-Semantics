# standard library imports
from argparse import ArgumentParser
import json
import sys
import logging
logger = logging.getLogger(__name__)

# third party imports
import pandas as pd
from huggingface_hub import login
from unsloth import FastLanguageModel
from tqdm import tqdm
import json_repair

# local imports
from json_utils import validate_json


class LLM:
    
    def __init__(self, 
                 model_name:str, 
                 prompt_path:str, 
                 instruct_path:str, 
                 hf_key_path:str):
        
        self.model, self.tokenizer = self.load_model(model_name)
        
        self.template = self.load_text(prompt_path)
        self.instruction = self.load_text(instruct_path)
        
        self.hf_key = self.load_text(hf_key_path)
        try:
            login(self.hf_key)
        except:
            raise KeyError('Please ensure HuggingFace key is valid')
        
    def load_model(self, model_name:str):
        """Loads the specified modle from Huggingface. Please use a model
        hosted on the Unsloth page.
        
        args:
            model_name (str) : path to model on huggingface e.g. 
            "unsloth/Llama-3.2-3B-Instruct"
        returns:
            FastLanguageModel : unsloth hosted model.
            Tokenizer : corresponding tokenizer for model.
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True)
        model = FastLanguageModel.for_inference(model)
        return model, tokenizer
    
    def load_text(self, path:str):
        """Loads data from a txt file
        
        args:
            path (str) : path to the text data.
        returns:
            str : the text.
        """
        with open(path, 'r') as f:
            text = f.read()
        return text
        
    def get_model_response(self, text:str, location:str, max_tokens=128, max_retries=5):
        """Passes the text to the LLM and returns a JSON
        args:
            text (str) : text to be processed.
            location (str) : location relavent to text (e.g. London)
            max_new_tokens (int) : max output size for model.
            max_retries (int) : max number of times to retry if json is broken
            curr_retries (int) : current number of retries performed.  
        returns
            Json formatted list[dict[str, str]]
        """
        for attempt in range(max_retries):
            prompt = self.template.format(self.instruction, text, "")
            prompt = prompt.replace('<location>', location)
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            response = self.model.generate(**inputs, max_new_tokens=max_tokens)
            output = self.tokenizer.decode(response[0])

            processed = self.process_output(output)

            if isinstance(processed, dict) and validate_json(processed):
                return processed  
            
            max_tokens += 32
        
        return output           
            
        
    def process_output(self, output:str)->list[dict[str,str]]:
        """Takes a string output from the LLM, extracts the relavent JSON and 
        processes it with reference to the schema defined in 'json_utils'
        
        args: 
            output (str) : a string containing a (possibly misconstructed) json.
        
        returns:
            list[dict[str,str]] : A valid json in accordance with the shema. 
        
        notes:
            Returns ['misconstructed json'] or ['invalid json'] if the json 
            could not be constructed or validated, respectively.
        """
        try:
            response = output.split("### Response")[1]
            json_block = response.split("<|startofjson|>")[1].split("<|endofjson|>")[0]
            cleaned_json = json_block.strip()
            return json_repair.loads(cleaned_json)
        except Exception as e:
            return None  
        

def main(data:pd.DataFrame, model:LLM, save_path:str):

    logging.basicConfig(filename='llama_log.log', level=logging.INFO)
    logger.info('Started')
    out = []
    
    for i, row in tqdm(data.iterrows(), total=len(data)):
        text = row.description
        location = 'Edinburgh'
        nearby_locations = model.get_model_response(text=text, location=location)
        
        out.append({'location':'Edinburgh',
                    'latitude':str(row.latitude),
                    'longitude':str(row.longitude),
                    'description':text,
                    'nearby':nearby_locations})
        
        if (i==1) or (i%25==0):
            with open(save_path, 'w') as f:
                json.dump(out, f)
                
    with open(save_path, 'w') as f:
                json.dump(out, f)
    logger.info('Finished')
        

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("-m", "--model", 
                        help="Name of unsloth model on huggingface",
                        )
    
    parser.add_argument("-p", "--prompt", 
                        help="Path to system prompt template",
                        )
    parser.add_argument("-i", "--instruct", 
                        help="Path to system instruction template",
                        )
    parser.add_argument("-k", "--key", 
                    help="Path to huggingface key",
                    )
    parser.add_argument("-d", "--data", 
                        help=" Path to data"
                        )
    parser.add_argument("-s", "--save", 
                        help="Path to save output json"
                        )

    
    args = parser.parse_args()
    
    #load data and model
    df = pd.read_csv(args.data)

    model = LLM(model_name=args.model, 
                prompt_path=args.prompt, 
                instruct_path=args.instruct,
                hf_key_path=args.key
                )
    # run main
    main(df, model, args.save)
    