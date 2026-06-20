#!/bin/bash
# ====================================================================
# deploy.sh — Telemetry Decoder Hugging Face Spaces Deployment Script
# ====================================================================

# Set color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================================="
echo "   Hugging Face Spaces Deployment Helper"
echo -e "====================================================================${NC}"

# Check for git
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git command line tool not found. Please install git.${NC}"
    exit 1
fi

echo -e "This script will help you deploy the FastAPI backend to Hugging Face Spaces."
echo -e "Before continuing, please make sure you have:"
echo -e " 1. Created a Hugging Face Account (https://huggingface.co)"
echo -e " 2. Created a new Space with the ${GREEN}Docker${NC} SDK option selected"
echo -e " 3. Obtained your Space's Git URL (e.g. https://huggingface.co/spaces/username/space-name)"
echo ""

# Ask for the Space Repository Git URL
read -p "Enter your Hugging Face Space Git repository URL: " HF_GIT_URL

if [ -z "$HF_GIT_URL" ]; then
    echo -e "${RED}Error: Repository URL cannot be empty.${NC}"
    exit 1
fi

# Create a temporary deploy directory
DEPLOY_DIR="deploy_temp"
echo -e "\n${BLUE}[1/4] Preparing deployment directory: ${DEPLOY_DIR}...${NC}"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy the server files and pipeline folder
echo -e "${BLUE}[2/4] Copying files to deployment package...${NC}"
cp server/app.py "$DEPLOY_DIR/"
cp server/Dockerfile "$DEPLOY_DIR/"
cp server_reference/requirements.txt "$DEPLOY_DIR/"
cp -R pipeline "$DEPLOY_DIR/"

# Navigate to deploy directory
cd "$DEPLOY_DIR" || exit

# Initialize git repository
echo -e "${BLUE}[3/4] Initializing temporary git repo & committing...${NC}"
git init -b main
git config user.name "Telemetry Deployer"
git config user.email "deploy@telemetry.local"
git add .
git commit -m "Deploy Telemetry Decoder Simulation Backend API"

# Add remote and push
echo -e "${BLUE}[4/4] Pushing to Hugging Face Spaces...${NC}"
echo -e "If prompted, enter your Hugging Face username and Access Token as the password."
git remote add origin "$HF_GIT_URL"
git push -f origin main

# Cleanup
cd ..
rm -rf "$DEPLOY_DIR"

echo -e "\n${GREEN}===================================================================="
echo "   Deployment Push Complete!"
echo -e "====================================================================${NC}"
echo -e "Check the Hugging Face Spaces UI to monitor the container build process."
echo -e "Once built, configure your frontend app.js API_BASE URL to call your Space."
