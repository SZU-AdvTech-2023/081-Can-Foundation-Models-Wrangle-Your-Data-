import utils.data_utils as data_utils
import utils.prompt_utils as prompt_utils
import utils.constants as constants
import httpx
import json
import asyncio

# from langchain.llms import Ollama

data_dir = '/home/dseg/cliu/FlexGen/flexgen/apps/data_wrangle/data/datasets/entity_matching/structured/Amazon-Google'


pd_data_files = data_utils.read_data(
    data_dir=data_dir,
    add_instruction=False,
    max_train_samples=-1,
    max_train_percent=-1,
    sep_tok='.',
    nan_tok='nan',
)

test_data = pd_data_files['test']
row = test_data.iloc[2]
serialized_r = row['text']

prefix_exs = prompt_utils.get_manual_prompt(data_dir, row)

query = (prefix_exs + '\n' + serialized_r).strip()


# llm = Ollama(
#     model='mistral',
#     stop=['\n'], 
# )

BASE = 'http://localhost:11434'

async def get(prompt: str, context = None, model = 'llama2'):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f'{BASE}/api/generate',
            json={'model': model, 'prompt': prompt, 'context': context}
        )
    return response.text

def parse(text: str):
    full_text = ''
    for line in text.splitlines():
        obj = json.loads(line)
        if not obj['done']:
            full_text += obj['response']
        else:
            with open('context.json', 'w') as fout:
                json.dump(obj, fout)
    return full_text

async def llm(prompt: str, context = None, model = 'llama2'):
    resp = await get(prompt, context, model)
    print(resp)
    return parse(resp)

context = [518,25580,29962,3532,14816,29903,29958,5299,829,14816,29903,6778,13,13,29902,626,2675,304,6222,263,848,2358,574,1847,3414,411,11148,3304,29906,29889,512,278,9508,29879,2446,29892,306,674,2367,777,6455,29889,3529,1101,278,6455,322,2367,596,1234,29889,29871,518,29914,25580,29962,13,7027,29991,306,29915,29885,7960,304,1371,366,411,596,848,2358,574,1847,3414,773,11148,3304,29906,29889,3529,3867,278,6455,310,278,9595,366,864,592,304,2189,29892,322,306,29915,645,437,590,1900,304,6985,366,29889]

async def main():
    model = 'llama2'
    # resp = await get(f'I am going to execute a data wrangling task with {model}. '
    #                  f'In the prompts next, I will give some examples. Please follow'
    #                  f' the examples and give your answer. ', model=model)
    # print(resp)
    # print(resp)
    task_instruction = constants.DATA2INSTRUCT[data_dir]
    prompt = lambda x: f"{task_instruction} {x}"
    prompted = prompt(serialized_r)
    print(prompted)
    pred = await llm(prompted, context=context, model=model)
    print(pred)

if __name__ == '__main__':
    asyncio.run(main())

