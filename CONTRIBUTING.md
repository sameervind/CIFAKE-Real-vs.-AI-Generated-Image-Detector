# Contributing to CIFAKE Real vs. AI Image Detector

Thank you for your interest in improving this project! We welcome community contributions, bug reports, and enhancements.

## Development Workflow

1. **Fork & Clone**
   ```bash
   git clone https://github.com/<your-username>/cifake-detector.git
   cd cifake-detector
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Branching Strategy**
   - Create a feature branch: `git checkout -b feature/your-feature-name`
   - Keep commits focused and provide clear, informative commit messages.

4. **Running Tests**
   - Ensure all tests pass before opening a Pull Request:
   ```bash
   pytest tests/ -v
   ```

5. **Submitting Changes**
   - Push your branch to GitHub.
   - Open a Pull Request against the `main` branch with a summary of changes and testing performed.
