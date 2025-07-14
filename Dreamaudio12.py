# streamlit_dream_audio_neuralmap_advanced.py

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
from PIL import Image
import io
import base64
import time

# ------------------ Memory Structures ------------------

class WorkingMemory:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.patterns = OrderedDict()

    def store(self, pattern):
        timestamp = datetime.now().timestamp()
        self.patterns[timestamp] = pattern
        if len(self.patterns) > self.capacity:
            self.patterns.popitem(last=False)

class EpisodicMemory:
    def __init__(self, compress=True):
        self.episodes = OrderedDict()
        self.current_episode = None
        self.compress = compress

    def create_episode(self, timestamp):
        self.current_episode = timestamp
        self.episodes[timestamp] = {'patterns': [], 'emotional_tags': []}

    def store_pattern(self, pattern, emotional_tag):
        if self.current_episode is None:
            self.create_episode(datetime.now())
        if self.compress:
            pattern = tuple(np.round(pattern, 2))
        self.episodes[self.current_episode]['patterns'].append(pattern)
        self.episodes[self.current_episode]['emotional_tags'].append(emotional_tag)

    def cleanup(self, max_episodes=200):
        while len(self.episodes) > max_episodes:
            self.episodes.popitem(last=False)

# ------------------ Neural Components ------------------

class PatternRecognizer:
    def __init__(self, decay=0.01):
        self.pattern_memory = {}
        self.decay = decay

    def reinforce(self, pattern):
        for k in list(self.pattern_memory.keys()):
            self.pattern_memory[k] *= (1 - self.decay)
            if self.pattern_memory[k] < 0.001:
                del self.pattern_memory[k]
        self.pattern_memory[pattern] = self.pattern_memory.get(pattern, 0.0) + 0.1

class CuriosityEngine:
    def __init__(self, base_threshold=0.15, adapt_rate=0.05):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.adapt_rate = adapt_rate
        self.novelty_history = []

    def evaluate_novelty(self, similarity):
        novelty = 1.0 - similarity
        self.novelty_history.append(novelty)
        if len(self.novelty_history) > 50:
            self.novelty_history.pop(0)
        self._adapt_threshold()
        return novelty

    def _adapt_threshold(self):
        if len(self.novelty_history) == 0:
            return
        avg_novelty = np.mean(self.novelty_history)
        self.current_threshold += self.adapt_rate * (avg_novelty - self.current_threshold)
        self.current_threshold = max(0.05, min(0.9, self.current_threshold))

    def should_grow(self, similarity):
        return self.evaluate_novelty(similarity) > self.current_threshold

class HybridNeuralUnit:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)
        self.age = 0
        self.usage_count = 0
        self.emotional_weight = 1.2
        self.connections = {}
        self.specialty = np.std(self.position)
        self.last_update = datetime.now()
        self.learning_rate = 0.05

    def cosine_similarity(self, input_pattern):
        pattern = np.array(input_pattern, dtype=np.float32)
        if len(pattern) != len(self.position):
            pattern = np.resize(pattern, len(self.position))
        norm_a = np.linalg.norm(self.position)
        norm_b = np.linalg.norm(pattern)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(self.position, pattern) / (norm_a * norm_b)

    def update(self, reward, input_pattern=None):
        self.usage_count += 1
        self.age = 0
        self.emotional_weight = 1.0 + reward * 1.5
        if input_pattern is not None and reward > 0:
            input_arr = np.array(input_pattern, dtype=np.float32)
            if len(input_arr) != len(self.position):
                input_arr = np.resize(input_arr, len(self.position))
            self.position = (1 - self.learning_rate) * self.position + self.learning_rate * input_arr
            self.last_update = datetime.now()

    def decay_connections(self):
        for k in list(self.connections.keys()):
            self.connections[k] *= 0.95
            if self.connections[k] < 0.01:
                del self.connections[k]

class IntuitiveNeuralNetwork:
    def __init__(self):
        self.units = []
        self.memory = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.recognizer = PatternRecognizer()
        self.curiosity = CuriosityEngine()

    def grow(self, input_pattern):
        new_unit = HybridNeuralUnit(input_pattern)
        self.units.append(new_unit)
        return new_unit

    def process_input(self, input_pattern, reward=0):
        if not self.units:
            return self.grow(input_pattern), 0.0
        similarities = [(u, u.cosine_similarity(input_pattern)) for u in self.units]
        best_unit, best_similarity = max(similarities, key=lambda x: x[1])

        if best_similarity < 0.25 or self.curiosity.should_grow(best_similarity):
            return self.grow(input_pattern), best_similarity

        best_unit.update(reward, input_pattern)
        best_unit.decay_connections()
        self.memory.store(input_pattern)
        self.episodic.store_pattern(input_pattern, 1.0 + best_similarity)
        self.episodic.cleanup()
        return best_unit, best_similarity

    def train_on_sequence(self, sequence, reward=0):
        self.recognizer.reinforce(tuple(sequence))
        _, sim = self.process_input(sequence, reward)
        return sim

# ------------------ Streamlit App ------------------

st.set_page_config(page_title="🎷 Dreaming from Music", layout="wide")
st.title("🎵 Dream Fractals: A Neural Dream from Music")

uploaded_file = st.file_uploader("Upload MP3/WAV file", type=["mp3", "wav"])
frame_limit = st.slider("Number of Dream Frames", 100, 1000, 500, step=50)

def random_color_by_emotion(emotion):
    # Map emotional weight (0 to ~3) to colors
    base_colors = [
        (0.9, 0.1, 0.1),  # red
        (0.1, 0.9, 0.1),  # green
        (0.1, 0.1, 0.9),  # blue
        (0.9, 0.9, 0.1),  # yellow
        (0.9, 0.1, 0.9),  # magenta
        (0.1, 0.9, 0.9),  # cyan
    ]
    idx = int((emotion * 3) % len(base_colors))
    return base_colors[idx]

def draw_shape(ax, shape_type, x, y, size, color, reward, pattern):
    if shape_type == 0:
        # Circle
        circle = plt.Circle((x, y), size, color=color, alpha=0.7)
        ax.add_patch(circle)
    elif shape_type == 1:
        # Square
        square = plt.Rectangle((x - size/2, y - size/2), size, size, color=color, alpha=0.6)
        ax.add_patch(square)
    elif shape_type == 2:
        # Triangle
        triangle = plt.Polygon([[x, y + size/1.5], [x - size, y - size/2], [x + size, y - size/2]], color=color, alpha=0.7)
        ax.add_patch(triangle)
    elif shape_type == 3:
        # Polygon (pentagon)
        n = 5
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        verts = np.array([(x + size * np.cos(a), y + size * np.sin(a)) for a in angles])
        polygon = plt.Polygon(verts, color=color, alpha=0.5)
        ax.add_patch(polygon)
    elif shape_type == 4:
        # Lattice/grid of dots
        for dx in np.linspace(x - size, x + size, 5):
            for dy in np.linspace(y - size, y + size, 5):
                ax.plot(dx, dy, 'o', color=color, alpha=0.4, markersize=2)
    else:
        # Spiral doodle
        t = np.linspace(0, 4 * np.pi, 100)
        r = size * (t / (4 * np.pi))
        xs = x + r * np.cos(t)
        ys = y + r * np.sin(t)
        ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1.5)

if uploaded_file is not None:
    st.audio(uploaded_file)
    y, sr = librosa.load(uploaded_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    rms = librosa.feature.rms(y=y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    net = IntuitiveNeuralNetwork()
    signature = []
    frames = []

    st.write("Generating dream... 🧠")
    col1, col2 = st.columns(2)
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    placeholder1 = col1.empty()
    placeholder2 = col2.empty()

    for i in range(min(S_db.shape[1], frame_limit)):
        pattern = S_db[:, i] / 80.0  # normalize roughly to [0,1]
        vol = rms[i] if i < len(rms) else 0.5
        harmony = chroma[:, i].mean() if i < chroma.shape[1] else 0.5
        reward = (vol + harmony) / 2.0

        sim = net.train_on_sequence(pattern, reward=reward)
        signature.append(pattern.tolist())

        ax1.clear()
        shape_type = int((reward * 10 + sim * 10 + i) % 6)
        color = random_color_by_emotion(reward * sim * 2)
        size = 0.5 + reward

        # Draw a shape at center with parameters
        draw_shape(ax1, shape_type, 0, 0, size, color, reward, pattern)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.axis('off')
        placeholder1.pyplot(fig1)

        # Neural map visualization
        ax2.clear()
        for unit in net.units:
            pos = unit.position[:2] if len(unit.position) >= 2 else np.random.rand(2)
            # Normalize pos to [0,1] range for display (simple min-max scaling)
            pos_min, pos_max = -2, 2
            x_pos = (pos[0] - pos_min) / (pos_max - pos_min)
            y_pos = (pos[1] - pos_min) / (pos_max - pos_min)
            color = random_color_by_emotion(unit.emotional_weight)
            ax2.plot(x_pos, y_pos, 'o', color=color, alpha=0.8, markersize=10)
            # Draw connections lightly
            for conn in unit.connections:
                if conn < len(net.units):
                    target = net.units[conn].position[:2]
                    tx = (target[0] - pos_min) / (pos_max - pos_min)
                    ty = (target[1] - pos_min) / (pos_max - pos_min)
                    ax2.plot([x_pos, tx], [y_pos, ty], color='gray', alpha=0.3)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title(f'Neurons: {len(net.units)}', fontsize=14)
        placeholder2.pyplot(fig2)

        # Save frame for GIF
        buf = io.BytesIO()
        fig1.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))
        time.sleep(0.01)

    # Save GIF and offer download
    if frames:
        gif_buf = io.BytesIO()
        frames[0].save(gif_buf, format='GIF', append_images=frames[1:], save_all=True, duration=50, loop=0)
        gif_b64 = base64.b64encode(gif_buf.getvalue()).decode()
        href = f'<a href="data:image/gif;base64,{gif_b64}" download="neural_dream.gif">Download Dream GIF</a>'
        st.markdown(href, unsafe_allow_html=True)

    with open("audio_dream_signature.json", "w") as f:
        json.dump({"patterns": signature[:50]}, f)

    st.success("\u2705 Dream sequence complete!")
    st.json({"Dream Signature Sample": signature[0]})

else:
    st.info("Please upload an MP3 or WAV file to start dreaming.")


