# CTCAE Vector Embeddings with SQLite

This project creates vector embeddings for CTCAE (Common Terminology Criteria for Adverse Events) terms and allows searching similar terms for clinical notes using semantic similarity. It uses:

1. **MedCPT** models for creating medical embeddings
2. **SQLite** for storing and retrieving embeddings
3. **GPT-4o-mini** for selecting the best match

## Features

- Creates embeddings for CTCAE terms and stores them in SQLite
- Performs vector similarity search to find relevant CTCAE terms for clinical notes
- Uses GPT-4o-mini to select the best match from the top results
- Supports interactive mode and command-line options

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a faster Python package installer and resolver.

#### Windows
```
install_uv.bat
```

#### Linux/macOS
```bash
chmod +x install_uv.sh
./install_uv.sh
```

2. Set your OpenAI API key:
   ```
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/macOS
   export OPENAI_API_KEY=your_api_key_here
   ```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install PyTorch with CUDA support (for GPU)
# Using uv (faster):
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy pandas transformers openai tqdm

# OR using standard pip:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas transformers openai tqdm
```

## Usage

### Initial Setup

First, set up the database with your CTCAE CSV file:

```bash
python sqlite_embedder.py --csv ctcae_v5.csv --setup
```

### Example with Sample Note

```bash
python sqlite_embedder.py --example
```

### Analyze a Clinical Note

```bash
python sqlite_embedder.py --note "Patient presents with fatigue and hemoglobin of 9.5 g/dL"
```

### Interactive Mode

```bash
python sqlite_embedder.py
```

## How It Works

1. **Database Setup**: 
   - Loads CTCAE terms from CSV
   - Creates embeddings using MedCPT-Article-Encoder
   - Stores terms and embeddings in SQLite

2. **Vector Search**:
   - Converts clinical notes to embeddings
   - Performs vector similarity search using cosine similarity
   - Returns top K matches

3. **Best Match Selection**:
   - Uses GPT-4o-mini to analyze the top matches
   - Selects the most relevant CTCAE term based on clinical context
   - Provides reasoning for the selection

## Requirements

- Python 3.8+
- PyTorch with CUDA support (for GPU acceleration)
- OpenAI API key for GPT-4o-mini access
- CTCAE v5 CSV file

## License

MIT