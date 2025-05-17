# Online-P2L

This is the codebase for the project Online-P2L [^longnote].

[^longnote]: Adapted from the codebase for [Prompt-to-Leaderboard](https://arxiv.org/pdf/2502.14855).

The training configs for OP2L experiments are located in [training_configs](./training_configs)

Code to simulate and pre-generate Geometric Buffer code is at [p2l/online_training/geometric_buffer.py](p2l/online_training/geometric_buffer.py)

Code to simulate and pre-generate Replay Buffer code is at [p2l/online_training/replay_buffer.py](p2l/online_training/replay_buffer.py)

## Abstract
The evaluation of large language models (LLMs) traditionally relies on static leaderboards, which cannot capture prompt-specific performance variations. Prompt-to-Leaderboard (P2L) addresses this limitation by creating prompt-specific rankings through a technique called "prompt-to-regression"~\cite{frick2025prompt}. P2L itself unlocks granular per-prompt evaluations, cost-optimal routing between LLMs, and automatic strength/weakness detection. However, P2L has a key limitation that prevents real-world deployability: P2L remains constrained by its fixed architecture, which cannot accommodate newly released models without complete retraining. Furthermore, P2L is trained on over 2 million rows of human preference data, with the dataset size constantly increasing with time. Therefore, re-training P2L to update the model is prohibitively expensive, and increasingly so. Unfortunately, naively training P2L with chronologically ordered data causes catastrophic forgetting due to temporal distribution shift, making this strategy ineffective. Therefore, we introduce Online P2L (OP2L), which enables continuous learning of human preferences through two strategic buffering approaches, Replay Buffer and our novel Geometric Buffer, that selectively manage training examples based on model familiarity, thus preventing catastrophic forgetting while preserving evaluation accuracy as new models enter the ecosystem. Importantly, both these algorithms only modify the training data as it is streamed in, and require no complex architecture or learning algorithm changes. We perform hyperparameter ablations on both the Replay Buffer and Geometric Buffer to explore the resulting model quality. We find both strategies perform significantly better than the chronological baseline, nearly matching oracle performance. With the replay buffer, we decrease the loss delta by 65.1\% between the OP2L model and the oracle fully re-trained P2L model compared to naive chronological training. Even better, with our novel Geometric Buffer, we decrease the loss delta by 78.1\%.

With Online P2L, we can realize much of the potential of the P2L framework. With our OP2L, we can continuously update the OP2L with new incoming data and new LLMs. In theory, this means we can:

1. Continuously provide granular prompt-level model leaderboards even with newly deployed models.
1. Hold the highest performance via a continuously updated router.
1. Continuous, granular, regression tracking of new model deployments
\end{enumerate}

Therefore, OP2L and the techniques presented in this paper are the first step towards continuous, up-to-date, fine-grained LLM evaluations that allow safe and auditable model deployments. Moreover, our approach devises a generalizable framework that can be applied beyond P2L, such as in recommendation systems, alignment pipelines, or reward model training.


## Table of Contents

- [Online-P2L](#Online-P2L)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
    - [Installing `uv`](#installing-uv)
    - [Serving P2L Setup](#serving-p2l-setup)
    - [Serving a Router Setup](#serving-a-router-setup)
    - [Training Setup](#training-setup)
  - [Serving P2L](#serving-p2l)
  - [Serving an OpenAI Compatible Router](#serving-an-openai-compatible-router)
    - [Example: serving a Bradley-Terry based cost-optimal router](#example-serving-a-bradley-terry-based-cost-optimal-router)
    - [Example: serving a Grounded RK based simple cost router](#example-serving-a-grounded-rk-based-simple-cost-router)
  - [Calling the OpenAI Compatible Router](#calling-the-openai-compatible-router)
  - [Training a P2L Model](#training-a-p2l-model)
  - [Training an Online P2L Model](#online-training)
  - [Inferencing a P2L Model](#inferencing-a-p2l-model)
  - [AutoEval Suite](#autoeval-suite)
    - [Params](#params)
  - [Online Evals](#auto-eval-for-online-training)



## Environment Setup

Setup instuctions will be shown using `uv`, however any package management system will work. All environments are native to Python 3.10, other versions are untested but may also work.

### Installing `uv`

If you like the sound of ~50x faster environment setup times, run the following to install `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env
```

To create a Python virtual environment run:

```bash
uv venv env --python 3.10
```

To activate said environment, run:

```bash
source env/bin/activate
```

### Serving P2L Setup

To serve a P2L model first run:

```bash
uv pip install -r serve_requirements.txt
```

### Serving a Router Setup

To serve a OpenAI compatible router, first run:

```bash
uv pip install -r route/requirements.txt
```

### Training Setup

To train a P2L model first run:

```bash
uv pip install -r train_requirements.txt
```

## Serving P2L

Before getting started, make sure you have followed the steps in [Serving Setup](#serving-p2l-setup).

`python p2l.endpoint` considers the following arguments:

| Option | Short Flag | Description |
|--------|-----------|-------------|
| `--help` | `-h` | Show this help message and exit. |
| `--model-path MODEL_PATH` | `-m MODEL_PATH` | Path to the model repository. |
| `--model-type MODEL_TYPE` | `-mt MODEL_TYPE` | Type of the model. |
| `--head-type HEAD_TYPE` | `-ht HEAD_TYPE` | Type of model head. |
| `--loss-type LOSS_TYPE` | `-lt LOSS_TYPE` | Type of the loss function. |
| `--api-key API_KEY` | `-a API_KEY` | API key for authorization. |
| `--host HOST` | `-H HOST` | Host to run the server on. |
| `--port PORT` | `-p PORT` | Port to run the server on. |
| `--reload, --no-reload` | - | Whether to reload the endpoint on detected code changes (requires workers to be set to 1). |
| `--workers WORKERS` | - | Number of endpoint workers (each will hold a model instance). |
| `--cuda, --no-cuda` | - | Flag to enable using a GPU to host the model. Flag is true by default. |

For example, to run p2el/Qwen2.5-7B-bag-01112025, which is a Qwen2 based "bag" model, which has head type `rk`, we would run:

```bash
python -m p2l.endpoint --model-path p2el/Qwen2.5-7B-bag-01112025 --model-type qwen2 --head-type rk --api-key <your-desired-api-key>
```

This code will host the model running on 1 worker and host 0.0.0.0 and port 10250 by default. Reload will be enabled meaning code changes will reload the endpoint. Note that by default the endpoint expects to load the model onto a GPU, however by specifying `--no-cuda` you can run this on CPU only, which may work for smaller P2L models.

Each P2L model has an associated model list, which specifices which model each index of the outputted coefficients corresponds to. Below is an example function to get this model list from the hosted endpoint:

```python
def get_p2l_endpoint_models(base_url: str, api_key: str) -> List[str]:

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    try:
        response = requests.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        result = response.json()
        return result["models"]

    except Exception as err:
        print(f"An error occurred: {err}")
```

Below is an example python function to query the P2L endpoint:

```python
def query_p2l_endpoint(
    prompt: list[str], base_url: str, api_key: str
) -> Dict[str, List]:

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    payload = {"prompt": prompt}

    try:
        response = requests.post(
            f"{base_url}/predict", headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        return result

    except Exception as err:

        raise err
```

Note that the input is a list of strings. This is NOT for  a batch of prompts, but rather for each turn in a coversation. For example, given a 2 turn conversation:

```
User: "hi!"
Assistant: "Hello"
User: "what's 1+1?"
```

The correct P2L input would be:

```python
["hi!", "what's 1+1?"]
```

## Serving an OpenAI Compatible Router

Serve an OpenAI compatible router with `python -m router.openai_server`. The available arguments are shown below.

| Option | Short Flag | Description |
|--------|-----------|-------------|
| `--help` | `-h` | Show this help message and exit. |
| `--config CONFIG` | `-c CONFIG` | Path to the configuration file. |
| `--router-type ROUTER_TYPE` | - | Type of the router to use. Avaible types are `bt-endpoint` and `bag-endpoint`.|
| `--router-model-name ROUTER_MODEL_NAME` | - | Name of the router model. |
| `--router-model-endpoint ROUTER_MODEL_ENDPOINT` | - | Endpoint URL for the router model. |
| `--router-api-key ROUTER_API_KEY` | - | API key for the router authentication. |
| `--cost-optimizer COST_OPTIMIZER` | - | Enable or configure cost optimization settings. Available types are `optimal-lp`, `simple-lp`, `strict`.|
| `--port PORT` | `-p PORT` | Port to run the server on. |
| `--host HOST` | - | Host to run the server on. |
| `--api-key API_KEY` | - | API key for authorization. |
| `--reload, --no-reload` | - | Whether to reload the endpoint on detected code changes (requires workers to be set to 1). |
| `--workers WORKERS` | - | Number of endpoint workers (each will hold a model instance). |

### Example: serving a Bradley-Terry based cost-optimal router

First, similar to above [above](#serving-p2l), we need to start serving a P2L model, this time Bradley-Terry based. To do this, let's run:

```bash
python -m p2l.endpoint --model-path p2el/Qwen2.5-7B-bt-01132025 --model-type qwen2 --head-type bt --api-key <your-desired-api-key>
```

Now, we need to configure a routing config file. This will specify the available models and inference details for the router.

For example, here is an example configuration that specifies Claude-3.5-Sonnet and GPT-4o:

```yaml
model_configs:
    claude-3-5-sonnet-20241022:
        api_key: <your-api-key>
        base_url: null
        cost: 9.3110239362
        max_tokens: 8192
        name: claude-3-5-sonnet-20241022
        system_prompt: null
        temp: 0.7
        top_p: 0.7
        type: anthropic

    gpt-4o-2024-05-13:
        api_key: <your-api-key>
        base_url: null
        cost: 12.3166873868
        name: gpt-4o-2024-05-13
        system_prompt: 'You are ChatGPT, a large language model trained by OpenAI, based
        on the GPT-4 architecture.

        Current date: 2025-01-06


        Image input capabilities: Enabled

        Personality: v2'
        temp: 0.7
        top_p: 1.0
        type: openai
```

Notice how the system prompt, temperature, and top_p are defined. These replicate how the models are served on Chatbot Arena. P2L is trained with the expectation that the models are running on this configuration. Therefore, for the most reliable results, we recommend sticking to the configs shown in [`example_config.yaml`](./route/example_config.yaml), though alternatives should still function well.

Additionally, we allow for adjustment of the `cost` parameter. One natural choice is just cost per output token, however more accuracte cost estimates are better. For example, the costs in [`example_config.yaml`](./route/example_config.yaml) are calculated to be proportional to the formula `cost_per_output_token * average_output_tokens_per_response`.

Now, lets assume we put the above config content into `config.yaml`. To start the OpenAI compatible router we would run:

```bash
python -m router.openai_server --config config.yaml --router-type bt-endpoint --router-model-endpoint http://0.0.0.0:10250 --router-api-key <your-api-key> --cost-optimizer optimal-lp --api-key <your-endpoint-api-key>
```

Let's break down what this command means:

- `--router-type bt-endpoint`: we are using a Bradley-Terry based P2L model hosted on an endpoint.
- `--router-model-endpoint http://0.0.0.0:10250`: this is where the router endpoint is, generally the default address will be this if you are running the routing server on the same machine running the P2L endpoint.
- `--cost-optimizer optimal-lp`: we are using cost routing using the optimal linear program detailed in Theorem 1 of the paper.

>**Note**: `optimal-lp` is only compatible with BT models, and `simple-lp` is only compatible with grounded RK (sometimes specified as bag) models.


### Example: serving a Grounded RK based simple cost router

P2L has a class of "Grounded RK" models. These models produces coefficents such that `0.0` represents the threshold for a "usable" answer. We can leverage this to cost route to maximize $P(\text{Not Bad})$... whatever that means exactly. Below we detail the steps to run this routing setup.

First, start up the P2L endpoint:

```bash
python -m p2l.endpoint --model-path p2el/Qwen2.5-7B-bag-01112025 --model-type qwen2 --head-type rk --api-key <your-desired-api-key>
```

Then start up the router server:

```bash
python -m router.openai_server --config config.yaml --router-type bag-endpoint --router-model-endpoint http://0.0.0.0:10250 --router-api-key <your-api-key> --cost-optimizer simple-lp --api-key <your-endpoint-api-key>
```

## Calling the OpenAI Compatible Router

As aptly named, the router server is OpenAI compatible. We can call it like any other OpenAI compatible model:

```python
from openai import OpenAI

client = OpenAI(
    base_url: "<your_router_endpoint_url>/v1",
    api_key: "<your_router_api_key>",
)

prompt = "what's 828913*1234?"

response = client.chat.completions.create(
    model="-", # This field is actually not used
    message=[{"role": "user", "content": prompt}],
    stream=True, # Router is compatible with and without streaming.
)
# Notice no temperature, top_p, or system prompt is set.
# This allows the router to use the default provided by the config file.
# If you do pass in these fields, they will override the config.
```

If we want to specify a cost budget, we need to do the following:

```python
response = client.chat.completions.create(
    model="-", # This field is actually not used
    message=[{"role": "user", "content": prompt}],
    stream=True, # Router is compatible with and without streaming.
    extra_body={"cost": <desired_cost>}
)
```

## Training a P2L Model

This codebase also contains the training code for P2L models. To train a P2L model, first set up a training config. The [`training_configs`](./training_configs/) directory has many examples.

To train run, for example:

```bash
deepspeed --num_gpus=8 p2l.train --config training_configs/<your_config>.yaml --no-eval --save-steps 512
```

## Inferencing a P2L Model

To quickly inference on a dataset using P2L, run:

```bash
python -m p2l.eval --model <p2l_model_name> --dataset <hf_dataset_path> --head-type <head_type> --model-type <qwen2_or_llama> --batch-size 2
```

This will work on any dataset of single turn prompts under the column name `prompt`.

## Online-Training

Our online training code is found in [p2l/online_training](./p2l/online_training).

Online training is a drop-in replacement for normal training:

```bash
deepspeed --num_gpus=8 p2l.online_training.online_train --config training_configs/<your_config>.yaml --checkpoint <prev_model_weights> --no-eval --save-steps 512
```


## AutoEval Suite

Our in-depth evaluation code can be run using `p2l.auto_evals`.

### Params

- **a. Model List Params**
    1. Either provide `--model_repo`, which has a `model_list.json` file.
    2. Or provide a local `--model_list_path` file.

- **b. Val Data**
    1. **Data is in JSONL format**:
        - Provide a local `--eval_path`.
        - If no path is provided, the program will look for an `eval_outputs.jsonl` file in the `--model_repo` on HF.
    2. **Data is in JSON format (checkpoint files)**:
        - Provide a local `--checkpoint_path`.
        - Or provide remote `--hf_checkpoint_repo` and `--hf_checkpoint_file`.

- **c. Output Directory**
    1. Provide a local `--output_dir` or a remote `--hf_output_dir`.
    2. Provide `--output_file_name`.

- **d. Train Data (Optional)**
    - Provide `--hf_train_dataset` or a local `--train_path`.

- **e. Arena Data (Optional)**
    - Provide a local `--arena_path` (CSV with model rankings).

- **f. Provide Model Info**
    1. `--loss_type` (e.g., `bt`, `bt_tie`, `rk`).
    2. `--model_type` (e.g., `p2l`, `marginal`, `arena`, `marginal-gt`).
    3. `--categories`.

- **g. Provide Types of Metrics**
    1. `--simple_metrics`, `--category_metrics`, `--rand_subset_metrics`, `--aggr_scale_subset_metrics`.
    2. Use `--metrics_to_inc` to filter out which of the above metrics to include.

- **h. Random Subset Params**
    1. `--rand_subset_sizes`: Specify subset sizes.
    2. `--rand_num_samples`: Specify the number of samples per random subset size.

- **i. Aggregation Subset Params**
    1. `--aggr_scale_subset_sizes`: Specify subset sizes.
    2. `--aggr_scale_num_samples`: Specify the number of samples per random subset size.
    3. `--aggr_scale_gt`: Specify whether to use `marginal-gt` or `arena` as ground truth for categories.

---

## Auto-Eval for Online Training

It is as simple as:

```bash
python -m p2l.eval_chrono -m <hf_model_path> -v validation_sets -ht <head_type> -lt <loss_type> -od output_dir_name
```

## Citation

```
@misc{frick2025onlinep2l,
      title={Online Prompt-to-Leaderboard}, 
      author={Connor Chen and Joseph Tennyson and Evan Frick},
      year={2025},
}

@misc{frick2025prompttoleaderboard,
      title={Prompt-to-Leaderboard}, 
      author={Evan Frick and Connor Chen and Joseph Tennyson and Tianle Li and Wei-Lin Chiang and Anastasios N. Angelopoulos and Ion Stoica},
      year={2025},
      eprint={2502.14855},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.14855}, 
}
```
