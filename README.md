## README.md

# StyleTTS2 - Advanced Text-to-Speech Synthesis

This project demonstrates the use of StyleTTS2, a cutting-edge text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis.

## Features

*   **Human-Level Synthesis**: Generates high-quality speech that closely matches human intonation and rhythm.
*   **Style Diffusion**: Utilizes diffusion models to capture and transfer speaking styles.
*   **Adversarial Training**: Improves synthesis quality and naturalness through adversarial learning.
*   **Seen and Unseen Speakers**: Capable of generating speech for both speakers seen during training and completely new, unseen speakers.
*   **Emotional Expressiveness**: Control over emotional tone and expressiveness using `embedding_scale`.
*   **Longform Narration**: Provides functionality for generating consistent longform audio.
*   **Style Transfer**: Transfer the style from one reference audio to new speech generation.
*   **Voice Cloning**: Record your own voice and use it as a reference to synthesize new speech in your style.

## Installation

To set up the project, follow these steps:

1.  **Clone the Repository**: Navigate to your desired directory and clone the `StyleTTS2` repository.

    ```bash
    git clone https://github.com/yl4579/StyleTTS2.git
    cd StyleTTS2
    ```

2.  **Install Dependencies**: Install the necessary Python packages. This project relies on `SoundFile`, `torchaudio`, `munch`, `torch`, `pydub`, `pyyaml`, `librosa`, `nltk`, `matplotlib`, `accelerate`, `transformers`, `phonemizer`, `einops`, `einops-exts`, `tqdm`, `typing-extensions`, and `monotonic_align`.

    ```bash
    pip install SoundFile torchaudio munch torch pydub pyyaml librosa nltk matplotlib accelerate transformers phonemizer einops einops-exts tqdm typing-extensions git+https://github.com/resemble-ai/monotonic_align.git
    ```

3.  **Install `espeak-ng`**: This is required by `phonemizer`.

    ```bash
    sudo apt-get install espeak-ng
    ```

4.  **Download Models**: Clone the pre-trained models and extract reference audio.

    ```bash
    git-lfs clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS
    mv StyleTTS2-LibriTTS/Models .
    mv StyleTTS2-LibriTTS/reference_audio.zip .
    unzip reference_audio.zip
    mv reference_audio Demo/reference_audio
    ```

5.  **Download NLTK `punkt` tokenizer**: This is used for text processing.

    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

Once installed, you can use the provided Python scripts and Jupyter Notebook cells to:

### 1. Load Models

The notebook loads various components of the StyleTTS2 model, including the text encoder, BERT encoder, predictor, decoder, style encoder, diffusion model, and ASR/F0/PLBERT models. Essential utility functions for preprocessing audio and inference are also defined.

### 2. Basic Synthesis

Generate speech from text using a reference audio to determine the speaking style. This works for both seen and unseen speakers.

```python
text = "'' StyleTTS 2 is a text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis. ''"
reference_dicts = {'696_92939': "Demo/reference_audio/696_92939_000016_000006.wav"}

for k, path in reference_dicts.items():
    ref_s = compute_style(path)
    wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
    # Play the audio (using IPython.display.Audio)
```

### 3. Speech Expressiveness

Control the emotional expressiveness of the generated speech by adjusting the `embedding_scale` parameter.

```python
ref_s = compute_style("Demo/reference_audio/1221-135767-0014.wav")
texts = {
    'Happy': "We are happy to invite you to join us on a journey to the past...",
    'Sad': "I am sorry to say that we have suffered a severe setback...",
    'Angry': "The field of astronomy is a joke!",
    'Surprised': "I can't believe it! You mean to tell me that you have discovered a new species..."
}

for k, v in texts.items():
    # embedding_scale=1 for less emotional, embedding_scale=2 for more emotional
    wav = inference(v, ref_s, diffusion_steps=10, alpha=0.3, beta=0.7, embedding_scale=2)
    # Play the audio
```

### 4. Longform Narration

Generate consistent long-form audio by chaining sentences and maintaining style consistency using a convex combination of previous and current styles.

```python
passage = "If the supply of fruit is greater than the family needs, it may be made a source of income..."
path = "Demo/reference_audio/696_92939_000016_000006.wav"
s_ref = compute_style(path)
sentences = passage.split('.')
wavs = []
s_prev = None
for text in sentences:
    if text.strip() == "": continue
    text += '.'
    wav, s_prev = LFinference(text, s_prev, s_ref, alpha=0.3, beta=0.9, t=0.7, diffusion_steps=10, embedding_scale=1.5)
    wavs.append(wav)
# Play np.concatenate(wavs)
```

### 5. Style Transfer

Transfer the style from a reference text's emotion to a different target text while using a base reference speaker.

```python
ref_texts = {
    'Happy': "We are happy to invite you to join us on a journey to the past...",
    'Sad': "I am sorry to say that we have suffered a severe setback...",
    'Angry': "The field of astronomy is a joke!",
    'Surprised': "I can't believe it! You mean to tell me that you have discovered a new species..."
}
path = "Demo/reference_audio/1221-135767-0014.wav"
s_ref = compute_style(path)
text = "Yea, his honourable worship is within, but he hath a godly minister or two with him, and likewise a leech."
for k, v in ref_texts.items():
    wav = STinference(text, s_ref, v, diffusion_steps=10, alpha=0.5, beta=0.9, embedding_scale=1.5)
    # Play the audio
```

### 6. Voice Cloning (Record Your Own Voice)

Record your own voice and then synthesize new text using your recorded voice's style.

```python
# Assumes a function `record()` is available to record audio.
# See the notebook for the full implementation of the `record` function.
# audio = record(sec=10) # Record 10 seconds of your voice

text = "'' text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis. ''"
reference_dicts = {'You': audio_file_path_from_recording}

for k, path in reference_dicts.items():
    ref_s = compute_style(path)
    wav = inference(text, ref_s, alpha=0.1, beta=0.5, diffusion_steps=10, embedding_scale=2)
    # Play the audio
```

## Credits

This project is based on the [StyleTTS2](https://github.com/yl4579/StyleTTS2) research by Yunjin Lee, Jaemin Cho, and Chang Liu.

## License

[Specify your license here, e.g., MIT License]
