import gradio as gr
from huggingface_hub import InferenceClient
import random

models = [
    "google/gemma-7b",
    "google/gemma-7b-it",
    "google/gemma-2b",
    "google/gemma-2b-it"
]

clients = []
for model in models:
    clients.append(InferenceClient(model))


def format_prompt(message, history):
    prompt = ""
    if history:
        for user_prompt, bot_response in history:
            prompt += f"<start_of_turn>user{user_prompt}<end_of_turn>"
            prompt += f"<start_of_turn>model{bot_response}"
    prompt += f"<start_of_turn>user{message}<end_of_turn><start_of_turn>model"
    return prompt


def chat_inf(system_prompt, prompt, history, client_choice, seed, temp, tokens, top_p, rep_p):
    client = clients[int(client_choice) - 1]
    if not history:
        history = []
        hist_len = 0
    if history:
        hist_len = len(history)
        print(hist_len)

    generate_kwargs = dict(
        temperature=temp,
        max_new_tokens=tokens,
        top_p=top_p,
        repetition_penalty=rep_p,
        do_sample=True,
        seed=seed,
    )
    formatted_prompt = format_prompt(f"{system_prompt}, {prompt}", history)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True,
                                    return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield [(prompt, output)]
    history.append((prompt, output))
    yield history


def clear_fn():
    return None


rand_val = random.randint(1, 1111111111111111)


def check_rand(inp, val):
    if inp is True:
        return gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, value=random.randint(1, 1111111111111111))
    else:
        return gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, value=int(val))


with gr.Blocks() as app:
    gr.HTML(
        """<center><h1 style='font-size:xx-large;'>Google Gemma Models</h1></center>""")
    with gr.Group():
        with gr.Row():
            client_choice = gr.Dropdown(label="Models", type='index', choices=[c for c in models], value=models[0],
                                        interactive=True)
    chat_b = gr.Chatbot(height=500)
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    rand = gr.Checkbox(label="Random Seed", value=True)
                    seed = gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, step=1, value=rand_val)
                    tokens = gr.Slider(label="Max new tokens", value=6400, minimum=0, maximum=8000, step=64,
                                       interactive=True, visible=True, info="The maximum number of tokens")
            with gr.Column(scale=1):
                with gr.Group():
                    temp = gr.Slider(label="Temperature", step=0.01, minimum=0.01, maximum=1.0, value=0.9)
                    top_p = gr.Slider(label="Top-P", step=0.01, minimum=0.01, maximum=1.0, value=0.9)
                    rep_p = gr.Slider(label="Repetition Penalty", step=0.1, minimum=0.1, maximum=2.0, value=1.0)

    with gr.Group():
        with gr.Row():
            with gr.Column(scale=3):
                sys_inp = gr.Textbox(label="System Prompt (optional)")
                inp = gr.Textbox(label="Prompt")
                with gr.Row():
                    btn = gr.Button("Chat")
                    stop_btn = gr.Button("Stop")
                    clear_btn = gr.Button("Clear")

    chat_sub = inp.submit(check_rand, [rand, seed], seed).then(chat_inf,
                                                               [sys_inp, inp, chat_b, client_choice, seed, temp, tokens,
                                                                top_p, rep_p], chat_b)
    go = btn.click(check_rand, [rand, seed], seed).then(chat_inf,
                                                        [sys_inp, inp, chat_b, client_choice, seed, temp, tokens, top_p,
                                                         rep_p], chat_b)
    stop_btn.click(None, None, None, cancels=[go, chat_sub])
    clear_btn.click(clear_fn, None, [chat_b])
app.queue(default_concurrency_limit=10).launch()
