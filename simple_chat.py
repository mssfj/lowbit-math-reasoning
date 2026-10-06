#!/usr/bin/env python3
"""Interact with the FineMath final checkpoint using Transformers."""

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


FINEMATH_MODEL = 'mssfj/qwen25-0.5b-finemath-4plus'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=FINEMATH_MODEL)
    parser.add_argument('--untie-embeddings', action='store_true',
                        help='Use separate input/output weights, as in this repository\'s pretraining')
    parser.add_argument('--prompt', help='Generate once instead of starting the interactive loop')
    parser.add_argument('--mode', choices=['plain', 'chat'], default='plain',
                        help='plain: text completion; chat: tokenizer chat template')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='0 for greedy decoding; positive values enable sampling')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature < 0:
        parser.error('max-new-tokens must be positive and temperature must be nonnegative')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    dtype = torch.float32
    if device == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f'Loading {args.model} on {device} ...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.mode == 'chat' and not tokenizer.chat_template:
        parser.error('This tokenizer has no chat template; use --mode plain')
    config = AutoConfig.from_pretrained(args.model)
    # Older HF exports incorrectly advertised tied weights for this untied model.
    if args.model == FINEMATH_MODEL or args.untie_embeddings:
        config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=dtype).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    history = []

    def generate(text):
        if args.mode == 'chat':
            messages = history + [{'role': 'user', 'content': text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = text
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        limit = getattr(model.config, 'max_position_embeddings', None)
        if limit and inputs.input_ids.shape[1] + args.max_new_tokens > limit:
            raise ValueError('Context is too long; use /reset or a shorter prompt')
        options = dict(max_new_tokens=args.max_new_tokens,
                       do_sample=args.temperature > 0,
                       pad_token_id=tokenizer.pad_token_id)
        if args.temperature > 0:
            options.update(temperature=args.temperature, top_p=0.9)
        with torch.inference_mode():
            output = model.generate(**inputs, **options)
        response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if args.mode == 'chat':
            history.extend([{'role': 'user', 'content': text}, {'role': 'assistant', 'content': response}])
        return response

    if args.prompt is not None:
        print(generate(args.prompt))
        return
    print('FineMath base model: plain mode completes text; chat mode may not follow instructions.')
    print('Commands: /exit, /reset. Plain mode treats each input independently.')
    while True:
        try:
            text = input('\nYou> ').strip()
            if text.lower() in ['/exit', 'exit', 'quit']:
                break
            if text == '/reset':
                history.clear()
                print('History cleared.')
            elif text:
                print('\nModel>', generate(text))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        except ValueError as error:
            print(error)


if __name__ == '__main__':
    main()
