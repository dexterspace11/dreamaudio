# streamlit_dream_audio_neuralmap_advanced.py
# Full script with persistent dream canvas (abstract painting style)

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

# This version draws dream shapes on a persistent canvas like an abstract painting
import json

def random_color():
    return np.random.rand(3,)

def draw_shape(ax, shape_type, x, y, size, color, alpha):
    if shape_type == 0:
        ax.add_patch(plt.Circle((x, y), size, color=color, alpha=alpha))
    elif shape_type == 1:
        ax.add_patch(plt.Rectangle((x - size/2, y - size/2), size, size, color=color, alpha=alpha))
    elif shape_type == 2:
        points = [(x, y+size), (x-size, y-size), (x+size, y-size)]
        ax.add_patch(plt.Polygon(points, color=color, alpha=alpha))
    elif shape_type == 3:
        theta = np.linspace(0, 2*np.pi, 6)
        r = size
        xs = x + r * np.cos(theta)
        ys = y + r * np.sin(theta)
        ax.fill(xs, ys, color=color, alpha=alpha)

st.set_page_config(page_title="Dream from Music", layout="wide")
st.title("🧠 Dreaming from Music — Abstract Neural Painting")

uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
frame_limit = st.slider("Dream Complexity (Frames)", 100, 1000, 500, step=100)

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

    st.write("Generating persistent abstract dream... 🧠")
    col1, col2 = st.columns(2)
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    placeholder1 = col1.empty()
    placeholder2 = col2.empty()

    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.axis('off')

    for i in range(min(S_db.shape[1], frame_limit)):
        pattern = S_db[:, i] / 80.0
        vol = rms[i] if i < len(rms) else 0.5
        harmony = chroma[:, i].mean() if i < chroma.shape[1] else 0.5
        reward = (vol + harmony) / 2.0

        sim = net.train_on_sequence(pattern, reward=reward)
        signature.append(pattern.tolist())

        shape_type = int((reward * 10 + sim * 10 + i) % 4)
        color = random_color()
        size = 0.1 + reward * 0.8
        alpha = 0.3 + sim * 0.5
        x, y = np.random.uniform(-2, 2), np.random.uniform(-2, 2)

        draw_shape(ax1, shape_type, x, y, size, color, alpha)
        placeholder1.pyplot(fig1)

        # Neural map visualization (unchanged)
        ax2.clear()
        for unit in net.units:
            pos = unit.position[:2] if len(unit.position) >= 2 else np.random.rand(2)
            x_pos = (pos[0] + 2) / 4
            y_pos = (pos[1] + 2) / 4
            ax2.plot(x_pos, y_pos, 'o', color=color, alpha=0.8, markersize=10)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title(f'Neurons: {len(net.units)}', fontsize=14)
        placeholder2.pyplot(fig2)
        time.sleep(0.01)

    st.success("Dream complete!")
    st.json({"Sample Signature": signature[0], "Neurons": len(net.units)})

