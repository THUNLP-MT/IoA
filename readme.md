<p align="center">
  <img src="assets/IoA_logo.png" alt="Illusion of Alignment logo" width="160">
</p>

# Illusion of Alignment

**Illusion of Alignment: Detecting Hidden Disagreement in Collaborative Dialogue**

An illusion of alignment occurs when a collaborative dialogue ends in apparent agreement while participants still differ on goals, assumptions, or execution plans. **IoA-Suite** makes hidden disagreement measurable through diagnostic multiple-choice questions. **IoA-Prober-8B** generates these questions from public profiles and the observable transcript.

[Model weights](https://huggingface.co/kmingl/IoA-Prober-8B) · [Dataset](#dataset) · [Prompts](prompts/)

## Dataset

**IoA-Suite** contains synthetic English collaborative dialogues with two participants, turn-level inner thoughts, and planted ground-truth misalignments.

Files are organized as `data/<split>/<domain>/<task_type>_<index>.json`. Each record contains:

- `profiles`: the meeting scenario, two participant profiles, and planted misalignment points. A participant's `private_state` is their **private agenda**, and `hidden_assumptions` are their **tacit assumptions**. Name, role, expertise, and background form the **public profile**.
- `dialogue`: the number of turns and ordered turns, each with its index, speaker, inner thought, and utterance. Only the speaker can access their inner thought during participant simulation.

The splits contain distinct synthesized dialogues. Synthesized personas are illustrative and should not be treated as representative of real populations or as verified beliefs of real individuals.

## Evaluation

Use Python 3.10+ and an OpenAI-compatible Chat Completions endpoint.

```bash
python -m pip install -r requirements.txt
cp config.example.json config.json

export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-endpoint.example/v1"

python evaluate.py --config config.json --data data/test --output results/ioa-prober --limit 1
python evaluate.py --config config.json --data data/test --output results/ioa-prober
```

Set each model name in `config.json`. The endpoint must serve or route the configured models. The paper uses:

| Component | Model | Temperature |
| --- | --- | --- |
| Prober | Model being evaluated | 0.7 |
| Participant simulator | Gemini-3-Flash (`gemini-3-flash-preview-thinking`) | 0.2 |
| Validity filter | GPT-5.4 (`gpt-5.4`) | 0.2 |

The simulator makes one successful call per participant, answering the entire question set. Keep the simulator and validity filter fixed when comparing probers. Each endpoint should enable the thinking mode used in the paper; provider-specific model aliases may differ.

For components served at different endpoints, use `PROBER_BASE_URL` / `PROBER_API_KEY`, `SIMULATOR_BASE_URL` / `SIMULATOR_API_KEY`, and `VALIDITY_FILTER_BASE_URL` / `VALIDITY_FILTER_API_KEY`. Each value falls back to the corresponding `OPENAI_` environment variable.

Provider-specific parameters can be added under `request_options` in the relevant component. For a Qwen3 server that accepts chat template settings:

```json
{
  "model": "IoA-Prober-8B",
  "temperature": 0.7,
  "request_options": {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": true}}
  }
}
```

`reasoning_effort`, `max_tokens`, `max_completion_tokens`, and `top_p` are also accepted in `request_options`. Use parameters supported by your provider. A `null` temperature omits that parameter when required by an endpoint; such a run does not use the paper's stated sampling temperature.

### Records and scores

`results/ioa-prober/records/<domain>/<case>.json` stores the generated questions, participant answers, validity decisions, per-question categories, P/R/F1, and the exact requests and response content. `run.json` records model settings and a fingerprint of the code and prompts. `summary.json` reports the mean per-dialogue P, R, and F1 on a 0–1 scale; the terminal displays percentages.

The same command resumes completed cases. Failed or incomplete responses are recorded as errors and retried when the command is rerun. A complete aggregate score is produced after every requested case succeeds. Changing the data, prompts, code, or endpoint settings requires a new output directory. `--workers` controls case concurrency (default: 4).

### Protocol

1. The **Prober** sees public profiles and utterances.
2. Each **participant simulator** sees its own full profile and inner thoughts, the other participant's public profile, and the shared transcript.
3. Different option choices establish divergence mechanically.
4. The **validity filter** checks meaningfulness, implicitness, and non-redundancy, then attributes accepted divergences to ground truth where possible. It receives full profiles, ground truth, utterances, and divergent questions with both answers.

Questions are assigned to **A** (accepted and attributed), **B** (accepted and unattributed), or **C** (non-divergent or rejected). For each dialogue:

$$P = \frac{|A|}{|A|+|C|},\qquad R = \frac{|\{m\in M(\tau):\exists(q,m)\in A\}|}{|M(\tau)|},\qquad F1=\frac{2PR}{P+R}.$$

Category B is excluded from both scores. Recall counts each ground-truth point once. A zero denominator gives a score of zero. Aggregate F1 is the mean of per-dialogue F1 values, rather than the harmonic mean of aggregate P and R.

The fixed prompts are in `prompts/` and follow the paper's Evaluation Prompts appendix. JSON field names follow the appendix schemas. The dynamic context implements the information access described in the Evaluation Protocol.

## Responsible use

IoA-Prober-8B is intended for symmetric collaboration. Obtain informed consent from every participant, disclose that the model infers unstated assumptions, give all participants access to generated questions and aggregated results, and require participant confirmation before acting on a suspected disagreement. A split in MCQ answers is an invitation to discuss, not a verified statement about someone's beliefs.

The model's [responsible-use license](https://huggingface.co/kmingl/IoA-Prober-8B/blob/main/LICENSE) prohibits surveillance, employment decisions, asymmetric monitoring, and adversarial elicitation. License terms must be supported by safeguards in the deployed application.

## License and citation

Code, documentation, and IoA-Suite data are licensed under [Apache-2.0](LICENSE).

IoA-Prober-8B uses the [IoA Responsible Use License 1.0](https://huggingface.co/kmingl/IoA-Prober-8B/blob/main/LICENSE), with the upstream Qwen3-8B Apache-2.0 notice preserved.

```bibtex
@misc{liu2026illusion,
  title={Illusion of Alignment: Detecting Hidden Disagreement in Collaborative Dialogue}, 
  author={Kaiming Liu and Fuwen Luo and Ziyue Wang and Jinrui Ju and Yuxuan Liu and Xuanyu Lei and Yunghwei Lai and Peng Li and Yang Liu},
  year={2026},
  eprint={2608.08210},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2608.08210}, 
}
```
