"""Gradio UI for the music generation demo."""

import gradio as gr
from generate import generate_music, image_to_music_prompt, SAMPLE_RATE

DURATION_MAP = {
    "5 seconds": 256,
    "10 seconds": 512,
    "20 seconds": 1024,
}


def text_to_music(prompt: str, duration: str):
    tokens = DURATION_MAP.get(duration, 256)
    filepath = generate_music(prompt, duration_tokens=tokens)
    return filepath


def image_to_music(image_description: str, duration: str):
    prompt = image_to_music_prompt(image_description)
    tokens = DURATION_MAP.get(duration, 256)
    filepath = generate_music(prompt, duration_tokens=tokens)
    return prompt, filepath


with gr.Blocks(title="Music to My Ears") as demo:
    gr.Markdown("# Music to My Ears\n### AI Music Generation — Super Bowl Halftime Hackathon")

    with gr.Tab("Text → Music"):
        prompt_input = gr.Textbox(
            label="Describe the music",
            placeholder="epic Super Bowl halftime show, energetic drums, brass section, crowd cheering",
            lines=2,
        )
        duration_input = gr.Radio(["5 seconds", "10 seconds", "20 seconds"], value="5 seconds", label="Duration")
        generate_btn = gr.Button("Generate", variant="primary")
        audio_output = gr.Audio(label="Generated Music", type="filepath")

        generate_btn.click(text_to_music, inputs=[prompt_input, duration_input], outputs=audio_output)

    with gr.Tab("Scene → Music"):
        gr.Markdown("Describe a scene, image, or moment — we'll turn it into music.")
        scene_input = gr.Textbox(
            label="Describe the scene",
            placeholder="A sunset over the ocean, waves crashing, seagulls in the distance",
            lines=2,
        )
        scene_duration = gr.Radio(["5 seconds", "10 seconds", "20 seconds"], value="5 seconds", label="Duration")
        scene_btn = gr.Button("Generate", variant="primary")
        scene_prompt_output = gr.Textbox(label="Generated music prompt", interactive=False)
        scene_audio_output = gr.Audio(label="Generated Music", type="filepath")

        scene_btn.click(image_to_music, inputs=[scene_input, scene_duration], outputs=[scene_prompt_output, scene_audio_output])

    gr.Markdown("---\n*Powered by Meta MusicGen via HuggingFace Transformers*")

if __name__ == "__main__":
    demo.launch()