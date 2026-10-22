# Chapter 1: overview

## What is an LLM

A large language model is a neural network trained to predict the next token
in a sequence of text. "Large" refers to both the parameter count (GPT-2 small
is 124M, GPT-3 is 175B) and the training data (GPT-3 saw about 300B tokens).

The architecture is the transformer. The model does not store the training
corpus; it is far smaller than the text it was trained on and learns patterns
from it, not the text itself.

## Decoder-only

The original 2017 transformer had two halves:

- **Encoder**: reads the whole input at once. Every token can attend to every
  other token, in both directions.
- **Decoder**: generates the output one token at a time. Its attention is
  *masked* so each token can only see the tokens before it. It also had a
  cross-attention layer that looked at the encoder's output.

GPT keeps only the decoder, and drops the cross-attention layer too, because
with no encoder there is nothing for it to attend to. BERT does the opposite
and keeps only the encoder.

The real difference between the two is the mask, not the task. An encoder
sees context on both sides; a decoder sees only the past. That mask is what
I'll implement in chapter 3.

## Pretraining objective

The objective is next-token prediction: given the tokens so far, output a
probability distribution over the next one. The training signal comes from
the text itself (the next token is already there), so no human labelling is
needed. This is called **self-supervised** learning.

Why does such a simple objective produce a model that can translate, answer
questions, and write code? Because the corpus is huge and varied, and
predicting the next token well across all of it is only possible if the model
picks up grammar, facts, and reasoning patterns. Every one of those reduces
prediction error. The text already contains translations, questions followed
by answers, and code, so doing those tasks is just a special case of
continuing the text. Nobody trained the model to translate; translation is
what good next-token prediction on a corpus containing translations looks
like. People call these abilities "emergent" because they were not explicitly
trained for, but the word is a label for the observation, not the explanation.

## Autoregressive

Generation is one token at a time: the model predicts a token, that token is
appended to the input, and the model runs again on the longer sequence. The
model's own output becomes its next input. This is why generating 100 tokens
means 100 forward passes.

## Pretraining vs fine-tuning

| | Pretraining | Fine-tuning |
|---|---|---|
| Data | huge, unlabelled, raw text | small, labelled, task-specific |
| Objective | next-token prediction | depends on the task |
| Cost | enormous (GPU-years) | cheap (minutes to hours) |
| Result | a **base model** that completes text | a model that does one job |

Pretraining produces a base model: it continues text well but does not follow
instructions or classify anything. Fine-tuning takes the base model and adapts
it. The book does two kinds:

- **Classification fine-tuning**: replace the output layer with a small
  classifier head (e.g. spam / not spam) and train on labelled examples.
- **Instruction fine-tuning**: train on instruction-and-response pairs so the
  model behaves like an assistant.

Because pretraining is so expensive, most practical work is fine-tuning a
base model someone else pretrained. The book pretrains only on a tiny text to
show the mechanism, then loads OpenAI's GPT-2 weights for the fine-tuning
chapters.

## Zero-shot and few-shot

A pretrained model can be used without fine-tuning by putting the task in the
prompt. Zero-shot: just describe the task. Few-shot: include a few examples
in the prompt. Neither changes the weights; the examples steer which patterns
the model uses.

## The book's three stages

| Stage | Contents | Chapters | Where it lives in this repo |
|---|---|---|---|
| 1. Building blocks | data preparation and sampling, attention, the GPT architecture | 2, 3, 4 | `gpt/data.py`, `gpt/attention.py`, `gpt/model.py` |
| 2. Pretraining | training loop, evaluation, loading pretrained weights | 5 | `gpt/train.py`, `gpt/generate.py`, `gpt/weights.py` |
| 3. Fine-tuning | classifier, personal assistant | 6, 7 | `notebooks/06_*`, `notebooks/07_*` |

## Terms to keep straight

- **Token**: the unit the model reads and predicts. Not a word; a word can be
  one or several tokens. Chapter 2.
- **Parameters / weights**: the numbers the model learns. N in the 6ND rule.
- **Tokens per parameter**: training tokens divided by parameters. GPT-3 was
  about 1.7; compute-optimal is around 20; modern small models go to 80+.
- **6ND**: training FLOPs is about 6 × parameters × tokens. Lets you estimate
  any training run before starting it.
