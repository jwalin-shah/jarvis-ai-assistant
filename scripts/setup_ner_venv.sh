#!/bin/bash
# Setup script for spaCy NER virtual environment
#
# This creates a separate Python environment for spaCy to avoid
# dependency conflicts with MLX and other JARVIS dependencies.
#
# Usage:
#   ./scripts/setup_ner_venv.sh
#
# The NER service will be installed at ~/.jarvis/ner_venv/

set -e

NER_VENV_PATH="${HOME}/.jarvis/ner_venv"
SPACY_MODEL="en_core_web_sm"

echo "=== JARVIS NER Service Setup ==="
echo ""

# Check if venv already exists
if [ -d "${NER_VENV_PATH}" ]; then
    echo "NER venv already exists at ${NER_VENV_PATH}"
    echo "To reinstall, first run: rm -rf ${NER_VENV_PATH}"
    exit 0
fi

# Create directory structure
mkdir -p "${HOME}/.jarvis"

# Create virtual environment
echo "Creating virtual environment at ${NER_VENV_PATH}..."
python3 -m venv "${NER_VENV_PATH}"

# Activate and install dependencies
echo "Installing spaCy..."
"${NER_VENV_PATH}/bin/pip" install --upgrade pip
"${NER_VENV_PATH}/bin/pip" install spacy

# Download spaCy model
echo "Downloading spaCy model: ${SPACY_MODEL}..."
"${NER_VENV_PATH}/bin/python" -m spacy download "${SPACY_MODEL}"

# Verify installation
echo ""
echo "Verifying installation..."
"${NER_VENV_PATH}/bin/python" -c "
import spacy
nlp = spacy.load('${SPACY_MODEL}')
doc = nlp('Hello, my name is Sarah and I work at Apple.')
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(f'Test entities: {entities}')
print('NER setup successful!')
"

echo ""
echo "=== Setup Complete ==="
echo "NER venv: ${NER_VENV_PATH}"
echo "spaCy model: ${SPACY_MODEL}"
echo ""
echo "To start the NER service:"
echo "  jarvis ner start"
echo ""
echo "Or manually:"
echo "  ${NER_VENV_PATH}/bin/python scripts/ner_server.py"
