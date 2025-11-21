#!/bin/bash

# ==============================================================================
# YOLOv10 Person Detection - First Run Setup Script
# ==============================================================================
# Automatically sets up the project after cloning from GitHub
# Usage: ./first-run.sh
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emoji support
CHECK="✅"
CROSS="❌"
ROCKET="🚀"
GEAR="⚙️"
PACKAGE="📦"
BOOK="📚"
LIGHTNING="⚡"
CHIP="🔧"
FIRE="🔥"
PARTY="🎉"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${MAGENTA}$1${NC} $2"
}

print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

print_error() {
    echo -e "${RED}${CROSS} $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# ==============================================================================
# System Checks
# ==============================================================================

check_system() {
    print_header "${CHIP} Checking System Requirements"
    
    # Check OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "Operating System: macOS"
        
        # Check for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            print_success "Apple Silicon detected (M1/M2/M3/M4)"
            DEVICE="mps"
        else
            print_info "Intel Mac detected"
            DEVICE="cpu"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Operating System: Linux"
        
        # Check for NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            print_success "NVIDIA GPU detected"
            DEVICE="cuda"
        else
            print_info "No NVIDIA GPU detected, using CPU"
            DEVICE="cpu"
        fi
    else
        print_warning "Unknown OS: $OSTYPE"
        DEVICE="cpu"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed!"
        echo "Please install Python 3.9 or higher from: https://www.python.org/downloads/"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python version: $PYTHON_VERSION"
    
    # Check Python version (need 3.9+)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        print_error "Python 3.9 or higher is required (you have $PYTHON_VERSION)"
        exit 1
    fi
    
    # Check disk space (need ~150GB for full COCO dataset)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/Gi//')
    else
        FREE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    fi
    
    print_info "Available disk space: ${FREE_SPACE}GB"
    
    if [ "$FREE_SPACE" -lt 150 ]; then
        print_warning "You have less than 150GB free. Full COCO dataset requires ~150GB"
        print_info "You can still use a subset or pre-processed dataset"
    fi
    
    echo ""
}

# ==============================================================================
# Configuration Setup
# ==============================================================================

setup_configuration() {
    print_header "${GEAR} Setting Up Configuration"
    
    if [ -f ".env" ]; then
        print_warning "Configuration file .env already exists"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Keeping existing .env file"
            return
        fi
    fi
    
    print_step "${GEAR}" "Creating .env configuration..."
    
    # Copy template
    cp .env.example .env
    
    # Auto-detect and set device
    if [[ "$OSTYPE" == "darwin"* ]] && [[ $(uname -m) == "arm64" ]]; then
        sed -i '' 's/DEVICE=mps/DEVICE=mps/' .env
        print_success "Configured for Apple Silicon (MPS)"
    elif command -v nvidia-smi &> /dev/null; then
        sed -i '' 's/DEVICE=mps/DEVICE=cuda/' .env 2>/dev/null || sed -i 's/DEVICE=mps/DEVICE=cuda/' .env
        print_success "Configured for NVIDIA GPU (CUDA)"
    else
        sed -i '' 's/DEVICE=mps/DEVICE=cpu/' .env 2>/dev/null || sed -i 's/DEVICE=mps/DEVICE=cpu/' .env
        print_success "Configured for CPU"
    fi
    
    print_success "Configuration file created: .env"
    echo ""
}

# ==============================================================================
# Virtual Environment Setup
# ==============================================================================

setup_virtualenv() {
    print_header "${PACKAGE} Setting Up Python Virtual Environment"
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "${PACKAGE}" "Removing old virtual environment..."
            rm -rf venv
        else
            print_info "Using existing virtual environment"
            return
        fi
    fi
    
    print_step "${PACKAGE}" "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
    
    echo ""
}

# ==============================================================================
# Dependencies Installation
# ==============================================================================

install_dependencies() {
    print_header "${BOOK} Installing Dependencies"
    
    print_step "${LIGHTNING}" "Activating virtual environment..."
    source venv/bin/activate
    
    print_step "${LIGHTNING}" "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    print_success "pip upgraded"
    
    print_step "${BOOK}" "Installing Python packages (this may take a few minutes)..."
    print_info "Installing PyTorch, Ultralytics YOLOv10, and other dependencies..."
    
    pip install -r requirements.txt
    
    print_success "All dependencies installed"
    echo ""
}

# ==============================================================================
# Dataset Check
# ==============================================================================

check_dataset() {
    print_header "${FIRE} Checking Dataset"
    
    if [ -d "datasets/coco_person/images/train" ] && [ -d "datasets/coco_person/images/val" ]; then
        TRAIN_COUNT=$(find datasets/coco_person/images/train -name "*.jpg" 2>/dev/null | wc -l)
        VAL_COUNT=$(find datasets/coco_person/images/val -name "*.jpg" 2>/dev/null | wc -l)
        
        if [ "$TRAIN_COUNT" -gt 0 ] && [ "$VAL_COUNT" -gt 0 ]; then
            print_success "Dataset found: $TRAIN_COUNT training images, $VAL_COUNT validation images"
            DATASET_READY=true
        else
            print_warning "Dataset folder exists but no images found"
            DATASET_READY=false
        fi
    else
        print_warning "Dataset not found"
        DATASET_READY=false
    fi
    
    if [ "$DATASET_READY" = false ]; then
        echo ""
        print_info "To prepare the dataset, run:"
        echo "  1. python download_coco.py    # Download COCO 2017 dataset (~19GB)"
        echo "  2. python prepare_dataset.py  # Extract person class"
        echo ""
    fi
    
    echo ""
}

# ==============================================================================
# System Test
# ==============================================================================

run_tests() {
    print_header "${ROCKET} Running System Tests"
    
    source venv/bin/activate
    
    print_step "${LIGHTNING}" "Testing installation..."
    
    if python test_setup.py; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed. Please check the output above."
        return 1
    fi
    
    echo ""
}

# ==============================================================================
# Final Instructions
# ==============================================================================

show_next_steps() {
    print_header "${PARTY} Setup Complete!"
    
    echo -e "${GREEN}Your YOLOv10 Person Detection project is ready to use!${NC}"
    echo ""
    
    print_info "Quick Start Commands:"
    echo ""
    echo -e "${CYAN}# Activate the virtual environment:${NC}"
    echo "  source venv/bin/activate"
    echo ""
    
    if [ "$DATASET_READY" = true ]; then
        echo -e "${CYAN}# Start training:${NC}"
        echo "  python train.py"
        echo ""
        echo -e "${CYAN}# Run inference after training:${NC}"
        echo "  python inference.py --source 0          # Webcam"
        echo "  python inference.py --source image.jpg  # Image"
        echo "  python inference.py --source video.mp4  # Video"
    else
        echo -e "${CYAN}# Prepare the dataset first:${NC}"
        echo "  python download_coco.py    # Download COCO dataset (~19GB)"
        echo "  python prepare_dataset.py  # Process person images"
        echo ""
        echo -e "${CYAN}# Then start training:${NC}"
        echo "  python train.py"
    fi
    
    echo ""
    echo -e "${CYAN}# Test your setup anytime:${NC}"
    echo "  python test_setup.py"
    echo ""
    echo -e "${CYAN}# View configuration:${NC}"
    echo "  python config.py"
    echo ""
    echo -e "${CYAN}# Edit settings:${NC}"
    echo "  nano .env    # or use your preferred editor"
    echo ""
    
    print_info "Documentation:"
    echo "  • README.md         - Full project documentation"
    echo "  • QUICK_START.md    - Quick start guide"
    echo "  • CONTRIBUTING.md   - Contribution guidelines"
    echo ""
    
    print_info "Need help?"
    echo "  • GitHub Issues: https://github.com/yourusername/yolo-person/issues"
    echo "  • Documentation: See README.md"
    echo ""
    
    echo -e "${GREEN}${PARTY} Happy training! ${PARTY}${NC}"
    echo ""
}

# ==============================================================================
# Main Installation Flow
# ==============================================================================

main() {
    clear
    
    print_header "${ROCKET} YOLOv10 Person Detection - First Run Setup"
    
    echo -e "${CYAN}This script will set up your development environment.${NC}"
    echo ""
    echo "Steps:"
    echo "  1. Check system requirements"
    echo "  2. Create configuration file (.env)"
    echo "  3. Set up Python virtual environment"
    echo "  4. Install dependencies"
    echo "  5. Check dataset availability"
    echo "  6. Run system tests"
    echo ""
    
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    
    # Run setup steps
    check_system
    setup_configuration
    setup_virtualenv
    install_dependencies
    check_dataset
    
    # Run tests
    if ! run_tests; then
        print_error "Setup completed with errors. Please review the output above."
        exit 1
    fi
    
    # Show next steps
    show_next_steps
    
    # Create a marker file to indicate first run is complete
    touch .first-run-complete
}

# ==============================================================================
# Script Entry Point
# ==============================================================================

# Check if already run
if [ -f ".first-run-complete" ]; then
    print_warning "First-run setup has already been completed."
    read -p "Do you want to run it again? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. To re-run setup, delete .first-run-complete file."
        exit 0
    fi
fi

# Run main installation
main
