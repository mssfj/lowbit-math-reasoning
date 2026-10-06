#!/usr/bin/env python3
"""Build a reproducible short-context OpenMathInstruct-2 SFT subset."""

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, snapshot_download
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

SOURCE = 'nvidia/OpenMathInstruct-2'
SOURCE_REV = '469216e3f46f4dacf476b382e192485ea51a143e'
TOKENIZER = 'mssfj/qwen25-0.5b-finemath-4plus'
TOKENIZER_REV = '0600e6f3b418e77d7323b8934451ac66d49a1185'
EMBEDDER = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDER_REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
BENCHMARKS = {
    'gsm8k': ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866'),
    'math500': ('HuggingFaceH4/MATH-500', '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'),
}
SYSTEM = 'Solve the math problem step by step. Put the final answer in \\boxed{...}.'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = text.replace('−', '-').replace('×', '*').replace('÷', '/')
    text = re.sub(r'\\(?:left|right|displaystyle|textstyle)\b', '', text)
    return ' '.join(re.findall(r'[a-z]+|\d+(?:\.\d+)?|[+*/=<>^-]', text))


def number_template(text):
    return re.sub(r'\d+(?:\.\d+)?', 'NUM', normalize(text))


def final_box(text):
    starts = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def normalize_answer(text):
    text = unicodedata.normalize('NFKC', text).strip().strip('$')
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    text = re.sub(r'\\(?:left|right)\b|\\[,!; ]', '', text)
    text = re.sub(r'\s+', '', text)
    # Treat digit grouping commas as formatting, keeping tuple commas.
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    return text


def messages(row):
    return [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': row['problem']},
        {'role': 'assistant', 'content': row['generated_solution']},
    ]


def load_inputs(cache):
    def get(repo, revision, kind, patterns):
        out = cache / repo.replace('/', '--')
        snapshot_download(repo, repo_type=kind, revision=revision,
                          allow_patterns=patterns, local_dir=out)
        return out
    source = get(SOURCE, SOURCE_REV, 'dataset', ['data/train_1M-*.parquet'])
    tokenizer = get(TOKENIZER, TOKENIZER_REV, 'model',
                    ['tokenizer*', 'chat_template.jinja', 'special_tokens_map.json',
                     'added_tokens.json', 'merges.txt', 'vocab.json', 'config.json'])
    embedding = get(EMBEDDER, EMBEDDER_REV, 'model',
                    ['*.json', '*.safetensors', '*.txt', '1_Pooling/*'])
    gsm = get(*BENCHMARKS['gsm8k'], 'dataset', ['main/test-*.parquet'])
    math = get(*BENCHMARKS['math500'], 'dataset', ['test.jsonl'])
    benchmark_rows = []
    for p in sorted((gsm / 'main').glob('test-*.parquet')):
        for i, row in enumerate(pq.read_table(p).to_pylist()):
            benchmark_rows.append({'id': f'gsm8k/test/{i}', 'problem': row['question']})
    for i, line in enumerate((math / 'test.jsonl').read_text().splitlines()):
        row = json.loads(line)
        benchmark_rows.append({'id': f'math500/test/{i}', 'problem': row['problem']})
    assert len(benchmark_rows) == 1819, len(benchmark_rows)
    return source, tokenizer, embedding, benchmark_rows


def read_candidates(source, seed, counts):
    best = {}
    for p in sorted((source / 'data').glob('train_1M-*.parquet')):
        offset = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=8192):
            for j, row in enumerate(batch.to_pylist()):
                counts['source_rows'] += 1
                row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                if not all(row.get(k) for k in ['problem', 'generated_solution', 'expected_answer']):
                    counts['empty_field'] += 1
                    continue
                if row['problem_source'] not in ['math', 'augmented_math', 'gsm8k', 'augmented_gsm8k']:
                    counts['unknown_source'] += 1
                    continue
                if '[asy]' in row['problem'] or re.search(r'<\|[^>]+\|>', row['problem']+row['generated_solution']):
                    counts['diagram_or_control_token'] += 1
                    continue
                box = final_box(row['generated_solution'])
                if box is None or normalize_answer(box) != normalize_answer(row['expected_answer']):
                    counts['missing_or_mismatched_final_box'] += 1
                    continue
                key = normalize(row['problem'])
                if not key:
                    counts['empty_normalized_problem'] += 1
                    continue
                row.update(source_file='data/'+p.name, source_row=offset+j,
                           problem_id=digest(key), problem_group_id=digest(number_template(row['problem'])))
                rank = (len(row['generated_solution']), digest(str(seed)+row['generated_solution']))
                if key in best:
                    counts['duplicate_normalized_problem'] += 1
                    old = best[key]
                    old_rank = (len(old['generated_solution']), digest(str(seed)+old['generated_solution']))
                    if rank >= old_rank:
                        continue
                best[key] = row
            offset += batch.num_rows
        print('Scanned',p.name,'unique candidates',len(best),flush=True)
    counts['unique_answer_checked_candidates'] = len(best)
    return list(best.values())


def lexical_filter(rows, benchmarks, counts, audit):
    exact = {normalize(b['problem']): b['id'] for b in benchmarks}
    templates = {number_template(b['problem']): b['id'] for b in benchmarks}
    normalized = list(exact)
    benchmark_ids = [exact[k] for k in normalized]
    keep = []
    for row in rows:
        key = normalize(row['problem'])
        template = number_template(row['problem'])
        reason, match, score = None, None, None
        if key in exact:
            reason, match = 'benchmark_normalized_exact', exact[key]
        elif template in templates:
            reason, match = 'benchmark_numeric_template_exact', templates[template]
        else:
            hit = process.extractOne(key, normalized, scorer=fuzz.ratio, score_cutoff=90)
            if hit:
                reason, match, score = 'benchmark_fuzzy_ratio_90', benchmark_ids[hit[2]], hit[1]
        if reason:
            counts[reason] += 1
            audit.append({'problem_id':row['problem_id'], 'reason':reason,
                          'benchmark_id':match, 'score':score})
        else:
            keep.append(row)
    return keep


def encode_lengths(rows, tokenizer, max_tokens, counts):
    keep = []
    for start in range(0, len(rows), 512):
        batch = rows[start:start+512]
        chats = [messages(r) for r in batch]
        ids = tokenizer.apply_chat_template(chats, tokenize=True, add_generation_prompt=False)
        for row, tokens in zip(batch, ids):
            if len(tokens) > max_tokens:
                counts['over_token_limit'] += 1
            else:
                row['token_count'] = len(tokens)
                keep.append(row)
        if start % 25600 == 0:
            print('Tokenized',min(start+512,len(rows)),'/',len(rows),flush=True)
    return keep


def deduplicate_templates(rows, seed, counts):
    keep = {}
    for row in sorted(rows, key=lambda r:digest(str(seed)+r['problem_id'])):
        key = row['problem_group_id']
        if key in keep:
            counts['duplicate_numeric_template'] += 1
        else:
            keep[key] = row
    return list(keep.values())


def semantic_filter(rows, benchmarks, embedding_path, counts, audit):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer(str(embedding_path), device=device)
    encoder.max_seq_length = 512
    texts = [r['problem'] for r in rows]
    counts['embedding_input_over_512_tokens'] = sum(
        len(x)>512 for x in encoder.tokenizer(texts, truncation=False)['input_ids'])
    b_embeddings = encoder.encode([b['problem'] for b in benchmarks], batch_size=128,
                                  normalize_embeddings=True, convert_to_tensor=True)
    embeddings = encoder.encode(texts, batch_size=128, normalize_embeddings=True,
                                convert_to_tensor=True, show_progress_bar=True)
    nearest, similarities = [], []
    for start in range(0,len(rows),2048):
        scores = embeddings[start:start+2048] @ b_embeddings.T
        values, indices = scores.max(dim=1)
        similarities.extend(values.cpu().tolist())
        nearest.extend(indices.cpu().tolist())
    keep = []
    for row, score, index in zip(rows, similarities, nearest):
        benchmark = benchmarks[index]
        raw_ratio = fuzz.ratio(normalize(row['problem']),normalize(benchmark['problem']))
        template_ratio = fuzz.ratio(number_template(row['problem']),number_template(benchmark['problem']))
        remove = score >= .94 or (score >= .90 and (raw_ratio >= 70 or template_ratio >= 85))
        if remove:
            counts['benchmark_semantic_similarity'] += 1
            audit.append({'problem_id':row['problem_id'], 'reason':'benchmark_semantic_similarity',
                          'benchmark_id':benchmark['id'], 'cosine':score,
                          'fuzzy_ratio':raw_ratio,'numeric_template_ratio':template_ratio})
        else:
            row['nearest_benchmark_cosine'] = score
            keep.append(row)
    return keep


def write_outputs(rows, args, counts, audit, tokenizer):
    total_math = round(args.total * args.math_ratio)
    val_math = round(args.validation * args.math_ratio)
    selected = {'train':[], 'validation':[]}
    for group, total, val in [('math',total_math,val_math),
                               ('gsm8k',args.total-total_math,args.validation-val_math)]:
        pool = sorted([r for r in rows if group in r['problem_source']],
                      key=lambda r:digest(str(args.seed)+r['problem_id']))
        if len(pool) < total:
            raise RuntimeError(f'Not enough {group} examples: {len(pool)} < {total}')
        pool = pool[:total]
        pool.sort(key=lambda r:digest('split'+str(args.seed)+r['problem_group_id']))
        selected['validation'].extend(pool[:val])
        selected['train'].extend(pool[val:])
    split_stats = {}
    for split, data in selected.items():
        data.sort(key=lambda r:digest('order'+str(args.seed)+r['problem_id']))
        for row in data:
            row['messages'] = messages(row)
            row['source_dataset'] = SOURCE
            row['source_revision'] = SOURCE_REV
            row['source_split'] = 'train_1M'
        values = [r['token_count'] for r in data]
        split_stats[split] = {'num_rows':len(data), 'sources':dict(Counter(r['problem_source'] for r in data)),
                             'tokens':sum(values),'min_tokens':min(values),'max_tokens':max(values),
                             'mean_tokens':float(np.mean(values)),
                             'max_nearest_benchmark_cosine':max(r['nearest_benchmark_cosine'] for r in data)}
        path = args.output/'data'/f'{split}-00000-of-00001.parquet'
        path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(data),path,compression='zstd')
    assert len(selected['train'])==args.total-args.validation
    assert len(selected['validation'])==args.validation
    for key in ['problem_id','problem_group_id']:
        assert not ({r[key] for r in selected['train']} & {r[key] for r in selected['validation']})
    # Independently retokenize every exported example to verify the saved count.
    for data in selected.values():
        for start in range(0,len(data),512):
            part=data[start:start+512]
            ids=tokenizer.apply_chat_template([r['messages'] for r in part],tokenize=True,
                                              add_generation_prompt=False)
            assert all(len(t)==r['token_count']<=args.max_tokens for t,r in zip(ids,part))
    benchmark_info={k:{'repo':v[0],'revision':v[1],'split':'test'} for k,v in BENCHMARKS.items()}
    stats={'source':{'repo':SOURCE,'revision':SOURCE_REV,'split':'train_1M'},
           'tokenizer':{'repo':TOKENIZER,'revision':TOKENIZER_REV,
                        'chat_template_sha256':digest(tokenizer.chat_template)},
           'embedding':{'repo':EMBEDDER,'revision':EMBEDDER_REV,'max_seq_length':512},
           'benchmarks':benchmark_info,'seed':args.seed,'max_tokens':args.max_tokens,
           'target_total':args.total,'math_ratio':args.math_ratio,'system_prompt':SYSTEM,
           'filter_counts':dict(counts),'splits':split_stats,
           'decontamination':{'fuzzy_ratio':90,'unconditional_cosine':.94,
                              'conditional_cosine':.90,'conditional_fuzzy_ratio':70,
                              'conditional_numeric_template_ratio':85},
           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stats['files']={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((args.output/'data').glob('*.parquet'))}
    (args.output/'selection_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    with (args.output/'decontamination_audit.jsonl').open('w') as f:
        for entry in audit:f.write(json.dumps(entry)+'\n')
    shutil.copyfile(__file__,args.output/'prepare_openmath_sft.py')
    print(json.dumps(stats,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('preprocessing/outputs/openmath-sft-50k'))
    parser.add_argument('--cache',type=Path,default=Path('/tmp/openmath-sft-sources'))
    parser.add_argument('--total',type=int,default=50000)
    parser.add_argument('--validation',type=int,default=1000)
    parser.add_argument('--math-ratio',type=float,default=.70)
    parser.add_argument('--max-tokens',type=int,default=2048)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--candidate-multiplier',type=float,default=2)
    args=parser.parse_args()
    if not 0 < args.validation < args.total or not 0 < args.math_ratio < 1:
        parser.error('Require 0 < validation < total and 0 < math-ratio < 1')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    source,tokenizer_dir,embedding,benchmarks=load_inputs(args.cache)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir)
    counts=Counter();audit=[]
    rows=read_candidates(source,args.seed,counts)
    rows=encode_lengths(rows,tokenizer,args.max_tokens,counts)
    rows=deduplicate_templates(rows,args.seed,counts)
    # A deterministic stratified candidate pool makes expensive similarity checks tractable.
    candidates=[]
    for group,ratio in [('math',args.math_ratio),('gsm8k',1-args.math_ratio)]:
        pool=sorted([r for r in rows if group in r['problem_source']],
                    key=lambda r:digest(str(args.seed)+r['problem_id']))
        limit=round(args.total*ratio*args.candidate_multiplier)
        counts['eligible_'+group]=len(pool)
        candidates.extend(pool[:limit])
    counts['similarity_candidate_pool']=len(candidates)
    print('Similarity candidate pool',len(candidates),flush=True)
    candidates=lexical_filter(candidates,benchmarks,counts,audit)
    candidates=semantic_filter(candidates,benchmarks,embedding,counts,audit)
    write_outputs(candidates,args,counts,audit,tokenizer)


if __name__=='__main__':main()
