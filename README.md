Qwen 3 Voice Stack on AMD Radeon (ROCm)
This repository provides a production-ready Docker configuration for running the Qwen 3 LLM and Qwen 3 ASR models on AMD Radeon GPUs using the ROCm ecosystem.

System Architecture
The project is split into two specialized Docker containers:

Qwen 3 ASR Container: Real-time, high-fidelity speech recognition.

Qwen 3 LLM Container: Advanced text generation and reasoning.

Technical Specifications [Fact]
OS: Arch Linux (Current rolling release).

GPU: AMD Radeon RX 9070 XT / 7900 XT.

Software Stack: Docker + optimized ROCm PyTorch environment.

Motivation: Radeon GPUs offer a superior price-to-VRAM ratio, making them a powerful alternative for scaling local AI solutions without NVIDIA-specific constraints.

Development Methodology (The "AI-Human Loop")
[Hypothesis / My Experience]: When configuring ROCm environments in Docker, LLMs often hallucinate driver paths or compilation flags if not grounded in real-time system data.

The Golden Rule for Success [Probability]:
To prevent AI drift, it is crucial to provide the AI assistant with terminal output from:

Docker Build logs (to fix dependency and environment conflicts).

Runtime logs (to address GPU passthrough and memory issues).

License
This project is licensed under the MIT License.
