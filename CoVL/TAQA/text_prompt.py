import torch
from transformers import BertTokenizer
def text_prompt(data):
    text_aug = [f"a photo of action quality score {{}}", 
    		f"a picture of action quality score {{}}", 
    		f"Teacher action quality score of {{}}", 
    		f"{{}}, an action quality score",
                f"{{}} this is an action quality score", 
                f"{{}}, a video of action quality score", 
                f"Playing action quality score of {{}}", 
                f"This is {{}} a video of teacher action",
                f"Playing a quality score of action, {{}}", 
                f"Doing a quality score of action, {{}}", 
                f"Look, the teacher action quality score is {{}}",
                f"Can you score the action quality of {{}}?", 
                f"Video classification of {{}}", 
                f"A video with an action quality score of {{}}",
                f"The male teacher action quality score is {{}}", 
                f"The female teacher action quality score is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)
    CLS = '[CLS]'
    SEP = '[SEP]'
    tokenizer = BertTokenizer.from_pretrained('./bert/')
    for ii, txt in enumerate(text_aug):
        four = []
        for c in data:
            sentence= txt.format(c)
            sentences= tokenizer.tokenize(sentence)
            sentences = [CLS]+sentences+[SEP]
            sentences_id = tokenizer.convert_tokens_to_ids(sentences)
            max_len = 77
            padded_sentences_id = sentences_id + [0] * (max_len - len(sentences_id))
            four.append(padded_sentences_id)
        four=torch.tensor(four).cuda()
        text_dict[ii]=four
    return num_text_aug, text_dict
